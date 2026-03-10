# dataset/

Данные для обучения.

| Путь | Содержимое |
|---|---|
| `videos/` | Исходные видео (.mov) по категориям |
| `frames/` | Предизвлечённые JPEG-кадры (генерируются автоматически при `uv run train`) |
| `ball_annotations.json` | Позиции мяча: cx, cy, visibility, source |
| `annotations.json` | Игровые события: net, bounce, hit |

Разметка видео — [webapp.md](/docs/webapp.md).