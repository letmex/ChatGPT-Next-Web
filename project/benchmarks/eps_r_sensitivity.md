# eps_r Sensitivity (HI/HII peaks and first-crack time)

Run:

```bash
python -m project.eps_r_sensitivity
```

This prints a markdown table for `eps_r = 1e-4, 1e-5, 1e-6` with:

- `HI_peak`
- `HII_peak`
- `first_crack_t`

The COMSOL-aligned default `eps_r = 1e-5` is now set in `project/config.py`.
