#!/usr/bin/env owl

#zoo "e7d8b1f6fbe1d12bb4a769d8736454b9" (* LoadImage   *)
#zoo "61b0221e0dd6771e278a10998396b027" (* Decode *)

open Owl
open Owl_types
open Neural 
open Neural.S

let get_input_data img_name = 
  LoadImage.(read_ppm img_name |> extend_dim)

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

let _ = 
  let nn  = Graph.load "vgg16_owl.network" in 
  let img = get_input_data "panda.ppm" in 
  let preds = Graph.model nn img in
  to_tuples preds
