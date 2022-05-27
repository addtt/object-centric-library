# Documentation for variants

The variants specification should be as follows:
```yaml
dataset_1:
  variant_1:
    ...
  variant_2:
    ...
dataset_2:
  variant_1:
    ...
  variant_2:
    ...
```

The default configs of the datasets `dataset_1` and `dataset_2` are expected to be available 
in corresponding `dataset/dataset_1.yaml` and `dataset/dataset_2.yaml` files.

For each variant, we can specify a base variant to inherit properties from.
If the variant's parent is `null` then the defaults are taken from the dataset's standard definition.

The `...` content above is itself a dictionary expressed in YAML and contains the updates to
be applied to the base dataset. These updates are applied recursively in case of nested variants, 
which we can achieve with the `parent` field.

Shortcuts:
- If the parent key is missing, `parent=null` is inferred.
- Some variant names (`crop`, `occlusion`, `object_color`, `object_style`, `object_shape`)
can be automatically filled with the appropriate transform and variant type, if not specified.
Defaults are defined in `_get_variant_defaults()` in `data/dataset_variants.py`.

Usage example:
```shell
# Standard Tetrominoes dataset (no variant specified)
python train_object_discovery.py [...] dataset=tetrominoes

# Apply cropping to Tetrominoes
python train_object_discovery.py [...] dataset=tetrominoes +dataset.variant=crop

# Standard CLEVR-6 dataset
python train_object_discovery.py [...] dataset=clevr +dataset.variant=6

# Only images with more than 6 objects
python train_object_discovery.py [...] dataset=clevr +dataset.variant=greater_than_6
```
where `[...]` is to be replaced with other arguments.
The `+` syntax in hydra will append the config key `dataset.variant` which is otherwise
missing by default.
