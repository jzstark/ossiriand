#!/usr/bin/env owl

#zoo "e7d8b1f6fbe1d12bb4a769d8736454b9" (* LoadImage   *)
#zoo "9428a62a31dbea75511882ab8218076f" (* InceptionV3 *)

open Owl
open Owl_types
open Neural 
open Neural.S

let get_input_data img_name = 
  LoadImage.(read_ppm img_name |> extend_dim |> normalise)

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
  let img_name = "panda1.ppm" in
  let nn  = Graph.load "inception_owl2.network" in 
  let img = get_input_data img_name in 
  let preds = Graph.model nn img in
  to_tuples preds


(* 
#use "inception.ml"
let labels_json   = to_json   ~top:5 labels 
let labels_tuples = to_tuples ~top:5 labels
*)