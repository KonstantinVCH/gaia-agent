"""
Разворачивает агента на Hugging Face Spaces одной командой: создаёт Space,
загружает файлы и (по желанию) прописывает секрет HF_TOKEN.

Заменяет шаги 1-3 ручной инструкции. Вместо кликанья по веб-интерфейсу:

    pip install huggingface_hub
    set HF_TOKEN=ваш_токен          # Windows; в bash: export HF_TOKEN=...
    python deploy.py --name gaia-agent

Про токен: он читается из переменной окружения на ВАШЕЙ машине и уходит только
в API Hugging Face. Скрипт его не печатает, никуда не пишет и не сохраняет —
специально: значение секрета не должно попадать ни в логи, ни в консоль.

Токену нужны права на запись (Write), иначе он не сможет создать Space:
https://huggingface.co/settings/tokens

Полезные флаги:
    --name gaia-agent        имя Space (обязательно)
    --private                создать приватным (для сдачи задания нужен ПУБЛИЧНЫЙ)
    --no-secret              не прописывать HF_TOKEN в секреты Space
    --model <id>             прописать секрет MODEL_ID
    --vision-model <id>      прописать секрет VISION_MODEL_ID
    --dry-run                показать, что было бы сделано, и выйти
"""

import argparse
import os
import sys
from pathlib import Path

# Файлы, которые нужны Space для работы. Тесты, инструкцию и сам deploy.py
# не загружаем — на Space они не нужны и только мусорят репозиторий.
FILES_TO_UPLOAD = ["app.py", "agent.py", "tools.py", "requirements.txt", "README.md"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Создать Space и загрузить в него агента.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", required=True, help="Имя Space, например gaia-agent.")
    parser.add_argument("--private", action="store_true", help="Создать приватным (для сдачи нужен публичный!).")
    parser.add_argument("--no-secret", action="store_true", help="Не прописывать HF_TOKEN в секреты Space.")
    parser.add_argument("--model", default=None, help="Прописать секрет MODEL_ID.")
    parser.add_argument("--vision-model", default=None, help="Прописать секрет VISION_MODEL_ID.")
    parser.add_argument("--dry-run", action="store_true", help="Ничего не делать, только показать план.")
    args = parser.parse_args()

    groq_key = os.getenv("GROQ_API_KEY")
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ОШИБКА: не задана переменная окружения HF_TOKEN.", file=sys.stderr)
        print("Токен нужен с правами Write: https://huggingface.co/settings/tokens", file=sys.stderr)
        print("  Windows cmd:  set HF_TOKEN=hf_...", file=sys.stderr)
        print("  PowerShell:   $env:HF_TOKEN=\"hf_...\"", file=sys.stderr)
        print("  bash:         export HF_TOKEN=hf_...", file=sys.stderr)
        print("", file=sys.stderr)
        print("HF_TOKEN нужен именно для создания Space (права Write).", file=sys.stderr)
        print("Если исчерпан лимит инференса HF — дополнительно задайте", file=sys.stderr)
        print("GROQ_API_KEY, и модель будет работать через Groq.", file=sys.stderr)
        return 1

    here = Path(__file__).parent
    missing = [f for f in FILES_TO_UPLOAD if not (here / f).exists()]
    if missing:
        print(f"ОШИБКА: в папке нет нужных файлов: {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ОШИБКА: не установлен huggingface_hub. Выполните: pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi(token=token)

    # Узнаём владельца токена, чтобы собрать repo_id и заодно проверить, что
    # токен вообще рабочий — до попытки что-либо создавать.
    try:
        me = api.whoami()
        owner = me["name"]
    except Exception as e:
        print(f"ОШИБКА: токен не принят Hugging Face ({e}).", file=sys.stderr)
        return 1

    repo_id = f"{owner}/{args.name}"
    space_url = f"https://huggingface.co/spaces/{repo_id}"

    print(f"Аккаунт:    {owner}")
    print(f"Space:      {repo_id}")
    print(f"Видимость:  {'приватный' if args.private else 'публичный'}")
    print(f"Файлы:      {', '.join(FILES_TO_UPLOAD)}")
    secrets = []
    if not args.no_secret:
        secrets.append("HF_TOKEN")
    # GROQ_API_KEY прописывается отдельно и обязательно: HF_TOKEN нужен Space
    # для входа пользователя, но если месячный лимит HF исчерпан, модель придёт
    # с Groq — и без этого секрета агент на Space останется без модели вовсе.
    if groq_key:
        secrets.append("GROQ_API_KEY")
    if args.model:
        secrets.append(f"MODEL_ID={args.model}")
    if args.vision_model:
        secrets.append(f"VISION_MODEL_ID={args.vision_model}")
    print(f"Секреты:    {', '.join(secrets) if secrets else 'не задаются'}")

    if args.private:
        print(
            "\nВНИМАНИЕ: задание курса требует ПУБЛИЧНЫЙ Space — "
            "проверяющий должен видеть код. Приватный не подойдёт для сдачи."
        )

    if args.dry_run:
        print("\n--dry-run: ничего не сделано.")
        return 0

    # 1. Создаём Space (exist_ok=True — повторный запуск обновит существующий,
    #    а не упадёт: обновлять код после правок хочется без пересоздания).
    print("\n[1/3] Создаю Space...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=args.private,
            exist_ok=True,
        )
        print(f"      готово: {space_url}")
    except Exception as e:
        print(f"ОШИБКА при создании Space: {e}", file=sys.stderr)
        print("Частая причина — у токена нет прав Write.", file=sys.stderr)
        return 1

    # 2. Загружаем файлы
    print("[2/3] Загружаю файлы...")
    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(here),
            allow_patterns=FILES_TO_UPLOAD,
            commit_message="Deploy GAIA agent",
        )
        print(f"      загружено файлов: {len(FILES_TO_UPLOAD)}")
    except Exception as e:
        print(f"ОШИБКА при загрузке файлов: {e}", file=sys.stderr)
        return 1

    # 3. Прописываем секреты
    print("[3/3] Прописываю секреты...")
    try:
        if groq_key:
            api.add_space_secret(repo_id=repo_id, key="GROQ_API_KEY", value=groq_key)
            print("      GROQ_API_KEY — прописан")

        if not args.no_secret:
            # Тот же токен передаём Space как секрет: агенту он нужен, чтобы
            # обращаться к Inference API. Значение нигде не печатается.
            api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=token)
            print("      HF_TOKEN — прописан")
        if args.model:
            api.add_space_secret(repo_id=repo_id, key="MODEL_ID", value=args.model)
            print(f"      MODEL_ID = {args.model}")
        if args.vision_model:
            api.add_space_secret(repo_id=repo_id, key="VISION_MODEL_ID", value=args.vision_model)
            print(f"      VISION_MODEL_ID = {args.vision_model}")
        if args.no_secret and not (args.model or args.vision_model):
            print("      пропущено (--no-secret)")
    except Exception as e:
        # Не выходим с ошибкой: файлы уже загружены, секрет можно дописать руками.
        print(f"      ПРЕДУПРЕЖДЕНИЕ: не удалось прописать секрет ({e}).")
        print(f"      Добавьте вручную: {space_url}/settings")

    print("\nГотово.")
    print(f"Space:  {space_url}")
    print("Сборка занимает пару минут. Когда статус станет Running — откройте вкладку App,")
    print("войдите кнопкой Sign in with Hugging Face и нажмите Run Evaluation.")
    if args.no_secret:
        print("\nНе забудьте: без секрета HF_TOKEN в настройках Space агент не сможет вызвать модель.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
