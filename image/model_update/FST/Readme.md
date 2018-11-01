# Prerequisit 

Install `hdf5_caml`:
1. get clone from https://github.com/vbrankov/hdf5-ocaml
2. Following Pierre's PR, change `write_bigarray` to `reawd_bigarray` at the begining of `read_float_genarray` function
3. Build and install -- and do NOT use it in `utop`

# Steps

0. Download ckpt files from [here](https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ). Put them into `checkpoint/` directory.
1. Change the model names in "tf_to_hdf5.py", and run it 6 times. 
2. Build: `dune build hdf5_to_owl.exe`
3. run `_build/default/hdf5_to_owl.exe`
