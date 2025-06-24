import os
import re
import SimpleITK as sitk

def extract_strict_number(filename):
    """
    提取文件名中“最后一组数字”，用作匹配依据。
    例如：
    - voxel dose mean123.mha -> 123
    - mask body123.mha       -> 123
    - mask body3.mha         -> 3
    """
    base = os.path.splitext(filename)[0]
    numbers = re.findall(r'\d+', base)
    return int(numbers[-1]) if numbers else None

def resample_to_reference(mask, reference_img, interpolator=sitk.sitkNearestNeighbor):
    """
    将掩膜重采样到参考图像空间。
    """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_img)
    resample.SetInterpolator(interpolator)
    resample.SetDefaultPixelValue(0)
    return resample.Execute(mask)

def batch_resample_and_multiply(image_folder, mask_folder, output_folder,
                                image_ext='.mha', mask_ext='.mha', output_prefix='voxel_dose_mean_'):
    """
    批量处理函数：
    - 根据文件名中的编号匹配图像和掩膜
    - 掩膜重采样后与图像点乘
    - 输出为 masked_编号.mha
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.endswith(image_ext)]
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith(mask_ext)]

    image_dict = {extract_strict_number(f): os.path.join(image_folder, f) for f in image_files}
    mask_dict = {extract_strict_number(f): os.path.join(mask_folder, f) for f in mask_files}

    matched_numbers = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

    for num in matched_numbers:
        image_path = image_dict[num]
        mask_path = mask_dict[num]

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        resampled_mask = resample_to_reference(mask, image)
        resampled_mask = sitk.Cast(resampled_mask, sitk.sitkFloat64)
        result = sitk.Multiply(image, resampled_mask)

        # 输出名：masked_编号.mha
        output_name = f"{output_prefix}{num}{image_ext}"
        output_path = os.path.join(output_folder, output_name)

        sitk.WriteImage(result, output_path)
        print(f"[完成] 编号 {num} → {output_path}")

    # 报告未匹配的项
    unmatched_images = set(image_dict.keys()) - set(mask_dict.keys())
    unmatched_masks = set(mask_dict.keys()) - set(image_dict.keys())

    for num in sorted(unmatched_images):
        print(f"[警告] 图像编号 {num} 无对应掩膜")
    for num in sorted(unmatched_masks):
        print(f"[警告] 掩膜编号 {num} 无对应图像")

if __name__ == "__main__":
    image_folder = r"D:\pyproject\Denoise for mc -3d unet\dataset\1e5"
    mask_folder = r"D:\pyproject\Unet Base\dataset\BodyMask"
    output_folder = r"D:\pyproject\Unet Base\dataset\masked1e5"

    batch_resample_and_multiply(image_folder, mask_folder, output_folder)
