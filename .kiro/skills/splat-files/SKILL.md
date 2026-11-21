---
name: splat-files
description: Field layout and value conventions of a standard 3DGS PLY, and the PlayCanvas splat-transform 3.8.0 command line for writing .sog. Use when reading or writing splat .ply files, choosing PLY column names, decoding scale/opacity/colour/spherical harmonics, or converting a trained splat to .sog or meta.json for the web viewer.
metadata:
  sources: PlayCanvas developer docs and the splat-transform v3.8.0 tag, read 2026
---

# Splat file formats

Two things: what a standard 3D Gaussian splat PLY holds, and how to turn one into a SOG
with PlayCanvas splat-transform 3.8.0. Everything below is from the PlayCanvas docs and the
`v3.8.0` tag of the splat-transform repo; links are at the bottom.

In our pipeline the PLY is the archive and interchange copy (full precision, large), and the
SOG is the small copy the web viewer loads. Both belong in `data/`, never in git.

## Standard 3DGS PLY

A splat PLY is an ordinary Stanford PLY, almost always `binary_little_endian 1.0`, with a
single `element vertex N` and no faces. Each vertex is one Gaussian, every splat field a
`float`, and there is no connectivity at all. Counts run from hundreds of thousands to
millions.

### Columns

| Column | Count | Holds |
| --- | --- | --- |
| `x`, `y`, `z` | 3 | Gaussian centre, in scene units |
| `scale_0`, `scale_1`, `scale_2` | 3 | per-axis size, log domain |
| `rot_0`, `rot_1`, `rot_2`, `rot_3` | 4 | orientation quaternion, `w` first |
| `opacity` | 1 | logit of the linear alpha |
| `f_dc_0`, `f_dc_1`, `f_dc_2` | 3 | band-0 (DC) spherical harmonic per colour channel |
| `f_rest_0` … | 0, 9, 24 or 45 | higher-order (AC) spherical harmonics |

Write `binary_little_endian`. splat-transform 3.8.0 parses the header as text and then reads
the body as fixed-size little-endian records; it has no ASCII parsing path.

Read by name from the header, never by fixed offset: producers order the columns differently.
PLYs from the original 3DGS lineage also carry zero `nx`, `ny`, `nz` normals, which nothing
renders. splat-transform's reader treats any column outside the list above as an extra
column, in file order.

A header `comment` can record how the scene was trained, and that changes how it must be
rendered: `comment SplatRenderMode: default | mip | 2dgs` (Brush, SplatTransform) or
`comment postshot.anti_aliasing=1` (Postshot). A 2DGS scene has `scale_0` and `scale_1` but no
`scale_2`; splat-transform reads that as 2DGS whatever the comments say, and fills the missing
axis with a zero thickness.

### Turning stored values into render values

```text
position  p = (x, y, z)
scale     s_i = exp(scale_i)
rotation  q = normalize((rot_0, rot_1, rot_2, rot_3))   # (w, x, y, z)
opacity   a = 1 / (1 + exp(-opacity))                   # sigmoid, 0..1
colour    c_i = 0.5 + 0.28209479177387814 * f_dc_i      # SH_C0 = 1 / (2*sqrt(pi))
```

Decoded colours are sRGB (sRGB transfer function, Rec.709 primaries) and display-referred:
no scene lighting, exposure or tone mapping applies, and blending happens in sRGB. Clamp the
evaluated colour at 0.

This stored-versus-linear split is also why splat-transform's `--filter-value` compares
transformed values for `opacity`, `scale_*` and `f_dc_*` by default, and wants an `_raw`
suffix (`opacity_raw`) to compare the numbers as they sit in the file.

### Spherical harmonics layout

`f_rest_*` is channel-major: all of channel R's coefficients, then G, then B.

| Bands beyond DC | Coefficients per channel | `f_rest_*` columns | R / G / B blocks |
| --- | --- | --- | --- |
| 0 | 0 | none | — |
| 1 | 3 | 9 | 0-2 / 3-5 / 6-8 |
| 2 | 8 | 24 | 0-7 / 8-15 / 16-23 |
| 3 | 15 | 45 | 0-14 / 15-29 / 30-44 |

So with the usual 45 columns, `f_rest_k` is channel `k / 15` and coefficient `k % 15`. Inside
a channel block the coefficients run band by band: band 1's 3, then band 2's 5, then band 3's
7. splat-transform rejects any other `f_rest_*` count.

The basis is the real SH basis with the Condon-Shortley phase, the same one gsplat uses, and
bands 1-3 match glTF's `KHR_gaussian_splatting`. Colour is evaluated against the unit vector
from the camera to the Gaussian centre, in the file's own coordinate frame. The SOG spec
§3.5.1 lists all the basis constants if you need to evaluate SH yourself.

### Coordinate frame

A 3DGS PLY is not in a +y-up world. To place one in a right-handed +y-up scene such as
PlayCanvas, rotate the whole scene 180° about z, `(x, y, z) -> (-x, -y, z)`, as the transform
of the node holding it. Rotations and SH coefficients stay defined in the file's frame, so
rotate the scene, not the per-Gaussian data.

### Reading PLYs from newer tools

The current SOG spec (version 3) lets a Gaussian carry opacity above 1, and says a converter
should then write a float `opacity_linear` column instead of `opacity`. If you see
`opacity_linear`, use it as the linear alpha directly and ignore `opacity`. splat-transform
3.8.0 neither writes nor reads that column.

## splat-transform 3.8.0

MIT licensed, needs Node 22 or newer. We use it as a command line tool only, which is the one
thing Node is in our stack for. Pin the version:

```bash
npm install -g @playcanvas/splat-transform@3.8.0
splat-transform --version
```

`npx -y @playcanvas/splat-transform@3.8.0 <args>` works too, if you would rather not install
globally.

### Command line shape

```text
splat-transform [OPTIONS] input [ACTIONS]  ...  output [ACTIONS]
```

Inputs become the working set, actions apply in the order written, the last file named is the
output, and actions after it modify the final result. `null` as the output discards the file.

### Writing a SOG

```bash
# bundled: one scene.sog file (a zip of meta.json plus the WebP textures)
splat-transform scene.ply scene.sog

# unbundled: a folder of meta.json plus the WebP textures
splat-transform scene.ply out/meta.json

# back to PLY, from either layout
splat-transform scene.sog roundtrip.ply
splat-transform out/meta.json roundtrip.ply
```

Options that apply when writing `.sog`, `meta.json`, `lod-meta.json` or `.html`:

| Option | Effect |
| --- | --- |
| `-i`, `--sh-iterations <n>` | iterations of SH compression, more is better quality. Default 10 |
| `--max-workers <n>` | worker threads for SOG encoding, 0 runs inline. Default 4 |
| `--webp-effort <0-9>` | lossless WebP effort: higher tries harder to shrink the textures |
| `-g <n\|cpu>` | device for GPU work: adapter index, or `cpu` |
| `--list-gpus` | list GPU adapters with their indices, then exit |
| `-w`, `--overwrite` | overwrite the output if it exists |
| `--stats [text\|json]` | per-column statistics and the fill/overdraw ratio to stdout |
| `--info [text\|json]` | format, per-LOD counts, extra columns, and the antialiased/2dgs tag |
| `-q` / `--verbose` | quieter, or debug-level diagnostics |

Actions worth knowing before a SOG write, since they decide how big it lands:

| Action | Effect |
| --- | --- |
| `-H`, `--filter-harmonics <0\|1\|2\|3>` | drop SH bands above n. `0` keeps DC colour only |
| `-N`, `--filter-nan` | drop Gaussians with NaN, most Inf, or an unrenderable rotation |
| `-V`, `--filter-value <name,cmp,value>` | keep where the column compares true, `cmp` in `lt lte gt gte eq neq` |
| `-d`, `--decimate <n\|n%>` | simplify to a count or a fraction |
| `-t` / `-r` / `-s` | translate `<x,y,z>`, rotate Euler degrees `<x,y,z>`, uniform scale `<factor>` |

So a cheap-to-load viewer copy, trimming SH to two bands and dropping bad Gaussians first:

```bash
splat-transform -w -i 20 scene.ply -N -H 2 scene.sog
```

`--decimate` has to be the final action and its output has to be `.ply`, so decimating and
writing a SOG takes two runs:

```bash
splat-transform scene.ply -d 25% scene-small.ply
splat-transform scene-small.ply scene-small.sog
```

Inspect without writing anything:

```bash
splat-transform scene.ply null --stats json
```

### GPU requirements

SOG compression uses the GPU through WebGPU, which on Linux runs over Vulkan, so the box needs
an NVIDIA driver that exposes a Vulkan ICD. g5 instances with an A10G work with a suitable
driver; no GRID driver is needed just for Vulkan. Check with `splat-transform --list-gpus`,
which should name the GPU, and `vulkaninfo --summary` on the host.

`-g cpu` always works and needs no driver, but SH compression runs roughly 5-10x slower. In
3.8.0 the GPU-only features, which `-g cpu` cannot do at all, are `--filter-cluster`,
`--filter-floaters`, `.voxel.json` output and `.webp` image rendering.

### What a SOG actually is

SOG (Spatially Ordered Gaussians) is a quantised, lossy container, usually 15-20x smaller than
the PLY. A bundled `.sog` is a zip of these files at the archive root; unbundled is the same
files in a folder:

| File | Holds |
| --- | --- |
| `meta.json` | version, count, codebooks, decode ranges, the filenames below |
| `means_l.webp`, `means_u.webp` | positions, 16 bits per axis across the two images, log domain |
| `scales.webp` | per-axis size as a codebook index, codebook in log domain |
| `quats.webp` | orientation, compressed quaternion |
| `sh0.webp` | DC colour as codebook indices in RGB, opacity in A |
| `shN_centroids.webp`, `shN_labels.webp` | SH palette and the per-Gaussian index into it (optional) |

Keep the textures lossless WebP. Lossy encoding corrupts the quantised values and shows up as
structural artefacts, not as soft blur.

**Version:** 3.8.0 writes `"version": 2` in `meta.json`, so read it against the SOG version 2
spec: opacity is `sh0.a / 255` capped at 1, SH is 1-3 bands, and the palette holds up to 65536
entries with 16-bit labels. Later splat-transform releases write version 3, which changes the
quaternion encoding, allows opacity above 1 and adds a fourth SH band, so pin the CLI version
and match it to what the viewer reads.

## Our rules that touch this

- splat-transform is a command line tool we install, not a vendored dependency, and Node
  exists in the stack for it alone.
- It needs its line in `docs/third-party-licences.md`, with its licence and how we use it, in
  the commit that adds it.
- PLYs, SOGs and anything else big live in `data/`. Never commit them.

## Links

- splat-transform README at the pinned version: <https://github.com/playcanvas/splat-transform/blob/v3.8.0/README.md>
- npm package: <https://www.npmjs.com/package/@playcanvas/splat-transform>
- PLY format: <https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/ply/>
- SOG version 3 spec, including the SH basis constants: <https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/>
- SOG version 2 spec, what 3.8.0 writes: <https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog-v2/>
- Running it on a Linux GPU box, driver and Vulkan setup: <https://developer.playcanvas.com/user-manual/splat-transform/docker/>
