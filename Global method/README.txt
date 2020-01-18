execution:
type in command line: python Global GenareteGlobal.py PARAMS

inputs:

params:
  -h, --help            show this help message and exit
  -i NIMAGES
                        num of images to process, default=4, 0<n<=4
  -p NPIXELS
                        ShadowPix size in pixels, default=300
  -a LIGHTANGLE
                        illumination angle in degrees, default=30, 0<n<90

  -v                    should print iteration data, default=True
  -c                    should use concurrency optimization, default=False, not recommended on windows

  -it NSTEPS
                        max num of simulated_annealing steps, default=10 ** 6,
                        each iteration take steps according to num of cores
  -s STOPAFTER
                        finish optimization if the global minimum candidate
                        remains the same for this number of steps,
                        default=10000
  -t TEMPINCLINE
                        reduce temp incline if the global minimum candidate
                        remains the same for this number of steps,
                        default=10000

input images should be saved in the same directory as the .py files
input filenames are from the pattern: img{direction}.jpg where direction is in ["+x", "+y", "-x", "-y"]
for less then 4 input images the first directions apply (e.g for two images use ["+x", "+y"])
images will be resized to the appropriate size


outputs:
once the program finish you will find these files in the directory:
res.stl - printable 3d shadowPix file
res_heightfield.npy - numpy file representing the heightfield
res_images.pdf - pdf showing the input images, simulated shadowPix and heightfield

every 1000'th iteration heightfield and pdf will be saved (so intermediate result could be recovered in case windows
 updates unexpectedly)