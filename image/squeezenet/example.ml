#!/usr/bin/env owl

open Owl
open Owl_types
open Neural 
open Neural.S

(* 1. Import Zoo Library *)
#zoo "e7d8b1f6fbe1d12bb4a769d8736454b9" (* LoadImage   *)
#zoo "41380a6baa6c5a37931cd375d494eb57" (* SqueezeNet  *)

let preprocess img = 
  let r = Dense.Ndarray.S.get_slice_simple [[];[];[];[0]] img in 
  let r = Dense.Ndarray.S.sub_scalar r 123.68 in 
  Dense.Ndarray.S.set_slice_simple [[];[];[];[0]] img r;

  let g = Dense.Ndarray.S.get_slice_simple [[];[];[];[1]] img in 
  let g = Dense.Ndarray.S.sub_scalar g 116.779 in 
  Dense.Ndarray.S.set_slice_simple [[];[];[];[1]] img g;

  let b = Dense.Ndarray.S.get_slice_simple [[];[];[];[2]] img in 
  let b = Dense.Ndarray.S.sub_scalar b 103.939 in 
  Dense.Ndarray.S.set_slice_simple [[];[];[];[2]] img b;
  img

let to_tuples ?(top=5) preds = 
  let dict_path = "imagenet1000.dict" in 
  let h = Owl_utils.marshal_from_file dict_path in 
  let tp = Dense.Matrix.S.top preds top in 

  let results = Array.make top ("type", 0.) in 
  Array.iteri (fun i x -> 
    let cls  = Hashtbl.find h x.(1) in 
    let prop = Dense.Ndarray.S.get preds [|x.(0); x.(1)|] in 
    Array.set results i (cls, prop);
  ) tp;
  results

let gist_id = ""

(* 2. Load Inception Network *)
let nn = Graph.load "sqnet_owl.network"

(* 3. Load Image *)
(* The image has to be of ppm format with size of 299x299.
 * One example image panda.ppm is included in this gist for you to try. *)

(* absolute path to your image *) 
let filename = "panda_sq.ppm"
let img = LoadImage.(read_ppm filename |> extend_dim) |> preprocess

(* 4. Image Classification *)
let preds = Graph.model nn img
let top = 5 (* default value *)
let response = to_tuples ~top preds
(* response is an array of tuples. Each tuple contains a category (string) and 
 * its inferred probability (float), ranging from 1 to 100. *)