Cityscapes setup notes

1. Download Cityscapes dataset (registration required):
   https://www.cityscapes-dataset.com/

2. Structure expected by this project (example):
   Cityscapes/
     leftImg8bit/
       train/
       val/
       test/
     gtFine/
       train/
       val/

3. Mapping and loader notes
   - A template mapping is provided at `config/class_map_cityscapes.yaml`.
   - The current `src/segmentation/class_map.py` expects an `ade20k` key
     in class specs. To use Cityscapes directly either:
     - copy `cityscapes` lists into `ade20k` keys in a local config file; or
     - extend `src/segmentation/class_map.py` to accept `cityscapes` keys.

4. Suggested next steps
   - Adapt `src/segmentation/mask2former.py` to load a Cityscapes-pretrained
     model if available, or use the existing ADE20K model as a baseline.
   - Run the pipeline with `--class-map config/class_map_cityscapes.yaml`.
