#!/usr/bin/env bash
set -e

OUTPUT_FILE="meg_workspace_summary.txt"
PASS_FILE="password_to_authorization_file.txt"

# 1. Проверяем файл и берем последнюю НЕПУСТУЮ строку
if [ ! -f "$PASS_FILE" ]; then
    echo "[-] Ошибка: Файл $PASS_FILE не найден!"
    exit 1
fi

# Игнорируем пустые строки в конце и берем реальный пароль
export HPC_PASS=$(grep -v '^[[:space:]]*$' "$PASS_FILE" | tail -n 1 | tr -d '\r\n')

if [ -z "$HPC_PASS" ]; then
    echo "[-] Ошибка: Не удалось извлечь пароль (получена пустая строка)!"
    exit 1
fi

echo "=== Сбор данных начат ==="
echo "[*] Пароль успешно считан (длина символов: ${#HPC_PASS})"

# 2. Локальный ls -R
echo "==========================================" > "$OUTPUT_FILE"
echo "LOCAL ls -R (meg_entropy_workspace)" >> "$OUTPUT_FILE"
echo "==========================================" >> "$OUTPUT_FILE"
ls -R >> "$OUTPUT_FILE"
echo "[+] Локальный ls -R сохранен"

# 3. Dockerfile
echo -e "\n==========================================" >> "$OUTPUT_FILE"
echo "FILE: Dockerfile" >> "$OUTPUT_FILE"
echo "==========================================" >> "$OUTPUT_FILE"
if [ -f "Dockerfile" ]; then
    cat "Dockerfile" >> "$OUTPUT_FILE"
else
    echo "[Dockerfile не найден]" >> "$OUTPUT_FILE"
fi
echo "[+] Dockerfile добавлен"

# 4. hpc
echo -e "\n==========================================" >> "$OUTPUT_FILE"
echo "FILE: hpc" >> "$OUTPUT_FILE"
echo "==========================================" >> "$OUTPUT_FILE"
if [ -f "hpc" ]; then
    cat "hpc" >> "$OUTPUT_FILE"
else
    echo "[hpc не найден]" >> "$OUTPUT_FILE"
fi
echo "[+] hpc добавлен"

# 5. week_00_test/Readme.md
echo -e "\n==========================================" >> "$OUTPUT_FILE"
echo "FILE: week_00_test/Readme.md" >> "$OUTPUT_FILE"
echo "==========================================" >> "$OUTPUT_FILE"
if [ -f "week_00_test/Readme.md" ]; then
    cat "week_00_test/Readme.md" >> "$OUTPUT_FILE"
elif [ -f "week_00_test/README.md" ]; then
    cat "week_00_test/README.md" >> "$OUTPUT_FILE"
else
    echo "[week_00_test/Readme.md не найден]" >> "$OUTPUT_FILE"
fi
echo "[+] week_00_test/Readme.md добавлен"

# 6. Подключение к cHARISMa через ./hpc connect и удаленный ls -R
echo -e "\n==========================================" >> "$OUTPUT_FILE"
echo "REMOTE ls -R (via ./hpc connect)" >> "$OUTPUT_FILE"
echo "==========================================" >> "$OUTPUT_FILE"

echo "[*] Подключаемся к HPC и выполняем ls -R..."

REMOTE_TMP=$(mktemp)

expect << 'EOF' > "$REMOTE_TMP"
set timeout 120
set pass $env(HPC_PASS)

# Вывод оставляем включенным, чтобы видеть реальный процесс
log_user 1

spawn ./hpc connect

# 1-й запрос пароля/passphrase
expect {
    -nocase -re "(passphrase|password|пароль).*:" {
        sleep 0.6
        send -- "$pass\r"
    }
    timeout {
        puts "\n[!] Таймаут ожидания 1-го запроса passphrase\n"
        exit 1
    }
    eof {
        puts "\n[!] Процесс завершился до 1-го ввода пароля\n"
        exit 1
    }
}

# 2-й запрос пароля/passphrase
expect {
    -nocase -re "(passphrase|password|пароль).*:" {
        sleep 0.6
        send -- "$pass\r"
    }
    timeout {
        puts "\n[!] Таймаут ожидания 2-го запроса passphrase\n"
        exit 1
    }
    eof {
        puts "\n[!] Процесс завершился после 1-го ввода (возможно, неверный пароль / Permission denied)\n"
        exit 1
    }
}

# Ждем приглашения командной строки кластера ($ или # или >)
expect {
    -re {([$#>]|\(base\)) *$} {
        sleep 0.5
        send -- "ls -R\r"
    }
    timeout {
        # Если приглашение нестандартное, отправляем команду через 2 секунды
        sleep 2
        send -- "ls -R\r"
    }
    eof {
        puts "\n[!] Сессия закрылась до появления командной строки\n"
        exit 1
    }
}

# Ждем окончания выполнения ls -R и возврата шелла
expect {
    -re {([$#>]|\(base\)) *$} {
        send -- "exit\r"
    }
    timeout {
        send -- "exit\r"
    }
}

expect eof
EOF

# Очищаем управляющие символы терминала и дописываем результат
sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" "$REMOTE_TMP" | tr -d '\r' >> "$OUTPUT_FILE"
rm -f "$REMOTE_TMP"

echo "[+] Удаленный ls -R успешно получен и добавлен!"
echo "=== Все данные записаны в $OUTPUT_FILE ==="
