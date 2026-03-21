# -*- coding: utf-8 -*-
"""
makeDatabase.py

Подготовка собственного датасета для лабораторной работы №5.
Скрипт:
1) читает сырые изображения из каталога raw_data/<class_name>;
2) находит область объекта (автомобиля) по отличию от белого фона;
3) делает кроп по объекту с отступами;
4) помещает объект на белый квадратный холст фиксированного размера;
5) делит изображения на train/test;

Ожидаемая структура исходных данных:
raw_data/
├─ mazda_mx5/
├─ corvette_c6/
├─ volkswagen_passat_b3/
└─ porsche_cayenne_1_rest/

Результат:
prepared_data/
├─ train/
│  ├─ mazda_mx5/
│  ├─ corvette_c6/
│  ├─ volkswagen_passat_b3/
│  └─ porsche_cayenne_1_rest/
└─ test/
   ├─ mazda_mx5/
   ├─ corvette_c6/
   ├─ volkswagen_passat_b3/
   └─ porsche_cayenne_1_rest/
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps


# -----------------------------
# Настройки по умолчанию
# -----------------------------
RAW_ROOT = Path("raw_data")
OUTPUT_ROOT = Path("prepared_data")

CLASS_NAMES = [
    "mazda_mx5",
    "corvette_c6",
    "volkswagen_passat_b3",
    "porsche_cayenne_1_rest",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Белый фон в jpg обычно не идеально белый, поэтому считаем фоном
# все почти белые пиксели.
BACKGROUND_THRESHOLD = 245

# Отступы вокруг найденного объекта (в пикселях исходного изображения)
OBJECT_PADDING = 12

# Размер итогового квадратного изображения
OUTPUT_SIZE = 512

# Дополнительная внутренняя рамка внутри квадратного холста,
# чтобы машина не прилипала к краям
INNER_MARGIN = 24

# Доля изображений, идущих в test
TEST_RATIO = 0.2

# Зерно генератора случайных чисел для воспроизводимости
RANDOM_SEED = 42

# Создавать ли простые аугментации для train
CREATE_AUGMENTATIONS = True


# -----------------------------
# Служебные функции
# -----------------------------
def list_images(folder: Path) -> List[Path]:
    """Возвращает список изображений из папки."""
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )



def ensure_dir(folder: Path) -> None:
    """Создает папку, если она не существует."""
    folder.mkdir(parents=True, exist_ok=True)



def crop_to_object(img: Image.Image,
                   bg_threshold: int = BACKGROUND_THRESHOLD,
                   padding: int = OBJECT_PADDING) -> Image.Image:
    """
    Находит ограничивающий прямоугольник объекта по отличию от белого фона
    и делает кроп с небольшим отступом.
    """
    rgb = img.convert("RGB")
    arr = np.array(rgb)

    # Маска: True там, где пиксель НЕ белый фон.
    # Берем пиксели, у которых хотя бы один канал заметно меньше порога.
    object_mask = np.any(arr < bg_threshold, axis=2)

    ys, xs = np.where(object_mask)

    # Если объект не найден, возвращаем исходное изображение
    if len(xs) == 0 or len(ys) == 0:
        return rgb

    left = max(int(xs.min()) - padding, 0)
    right = min(int(xs.max()) + padding + 1, rgb.width)
    top = max(int(ys.min()) - padding, 0)
    bottom = min(int(ys.max()) + padding + 1, rgb.height)

    return rgb.crop((left, top, right, bottom))



def paste_on_white_square(img: Image.Image,
                          output_size: int = OUTPUT_SIZE,
                          inner_margin: int = INNER_MARGIN) -> Image.Image:
    """
    Масштабирует объект с сохранением пропорций и помещает его
    на белый квадратный холст фиксированного размера.
    """
    img = img.convert("RGB")
    w, h = img.size

    max_w = output_size - 2 * inner_margin
    max_h = output_size - 2 * inner_margin
    if max_w <= 0 or max_h <= 0:
        raise ValueError("INNER_MARGIN слишком большой относительно OUTPUT_SIZE.")

    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (output_size, output_size), (255, 255, 255))
    x = (output_size - new_w) // 2
    y = (output_size - new_h) // 2
    canvas.paste(resized, (x, y))

    return canvas



def prepare_single_image(path: Path) -> Image.Image:
    """
    Полный конвейер подготовки одного изображения:
    открыть -> кроп по объекту -> квадратный белый холст.
    """
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        cropped = crop_to_object(rgb)
        prepared = paste_on_white_square(cropped)
    return prepared



def save_image(img: Image.Image, path: Path) -> None:
    """Сохраняет изображение в JPG."""
    ensure_dir(path.parent)
    img.save(path, format="JPEG", quality=95)



def shift_image(img: Image.Image, dx: int = 0, dy: int = 0) -> Image.Image:
    """
    Сдвигает изображение на белом холсте.
    """
    src = img.convert("RGB")
    arr = np.array(src)

    object_mask = np.any(arr < BACKGROUND_THRESHOLD, axis=2)
    ys, xs = np.where(object_mask)

    if len(xs) == 0 or len(ys) == 0:
        return src.copy()

    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1

    obj = src.crop((left, top, right, bottom))
    canvas = Image.new("RGB", src.size, (255, 255, 255))

    x0 = (src.width - obj.width) // 2 + dx
    y0 = (src.height - obj.height) // 2 + dy

    x0 = max(0, min(x0, src.width - obj.width))
    y0 = max(0, min(y0, src.height - obj.height))

    canvas.paste(obj, (x0, y0))
    return canvas



def rotate_on_white(img: Image.Image, angle: float = 3) -> Image.Image:
    """Поворачивает изображение на белом фоне."""
    return img.convert("RGB").rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )



def make_augmented_versions(img: Image.Image) -> List[Tuple[str, Image.Image]]:
    """
    Создает несколько простых аугментаций для train:
    - зеркальное отражение;
    - небольшой сдвиг;
    - небольшой поворот.
    """
    result: List[Tuple[str, Image.Image]] = []
    result.append(("flip", ImageOps.mirror(img)))
    result.append(("shift", shift_image(img, dx=14, dy=-8)))
    result.append(("rot", rotate_on_white(img, angle=3)))
    return result



def split_train_test(paths: List[Path], test_ratio: float = TEST_RATIO) -> Tuple[List[Path], List[Path]]:
    """Делит список путей на train/test."""
    shuffled = paths[:]
    random.shuffle(shuffled)

    test_count = max(1, int(round(len(shuffled) * test_ratio))) if len(shuffled) > 1 else 0
    test_paths = shuffled[:test_count]
    train_paths = shuffled[test_count:]

    if len(shuffled) == 1:
        train_paths = shuffled
        test_paths = []

    return train_paths, test_paths



def process_class(class_name: str) -> None:
    """Обрабатывает один класс целиком."""
    source_dir = RAW_ROOT / class_name
    images = list_images(source_dir)

    if not images:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Для класса '{class_name}' не найдено изображений в {source_dir}")
        return

    train_paths, test_paths = split_train_test(images)

    print(f"\nКласс: {class_name}")
    print(f"Всего изображений: {len(images)}")
    print(f"Train: {len(train_paths)} | Test: {len(test_paths)}")

    for idx, img_path in enumerate(train_paths, start=1):
        prepared = prepare_single_image(img_path)
        stem = img_path.stem
        out_path = OUTPUT_ROOT / "train" / class_name / f"{stem}.jpg"
        save_image(prepared, out_path)

        if CREATE_AUGMENTATIONS:
            for suffix, aug_img in make_augmented_versions(prepared):
                aug_path = OUTPUT_ROOT / "train" / class_name / f"{stem}_{suffix}.jpg"
                save_image(aug_img, aug_path)

        if idx % 10 == 0 or idx == len(train_paths):
            print(f"  train: обработано {idx}/{len(train_paths)}")

    for idx, img_path in enumerate(test_paths, start=1):
        prepared = prepare_single_image(img_path)
        stem = img_path.stem
        out_path = OUTPUT_ROOT / "test" / class_name / f"{stem}.jpg"
        save_image(prepared, out_path)

        if idx % 10 == 0 or idx == len(test_paths):
            print(f"  test : обработано {idx}/{len(test_paths)}")



def clean_output_root() -> None:
    """Создает базовую структуру папок результата."""
    for subset in ("train", "test"):
        for class_name in CLASS_NAMES:
            ensure_dir(OUTPUT_ROOT / subset / class_name)



def main() -> None:
    random.seed(RANDOM_SEED)

    print("Подготовка датасета начата...")
    print(f"Исходная папка: {RAW_ROOT.resolve()}")
    print(f"Выходная папка: {OUTPUT_ROOT.resolve()}")
    print(f"Классы: {CLASS_NAMES}")
    print(f"Размер итогового изображения: {OUTPUT_SIZE}x{OUTPUT_SIZE}")
    print(f"Порог фона: {BACKGROUND_THRESHOLD}")
    print(f"Padding объекта: {OBJECT_PADDING}")
    print(f"Доля test: {TEST_RATIO}")
    print(f"Аугментации train: {'включены' if CREATE_AUGMENTATIONS else 'выключены'}")

    clean_output_root()

    for class_name in CLASS_NAMES:
        process_class(class_name)

    print("\nГотово.")
    print("Подготовленный датасет сохранен в папке prepared_data")


if __name__ == "__main__":
    main()
