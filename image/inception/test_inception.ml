#!/usr/bin/env owl

#zoo "e7d8b1f6fbe1d12bb4a769d8736454b9" (* LoadImage   *)
#zoo "9428a62a31dbea75511882ab8218076f" (* InceptionV3 *)
#zoo "61b0221e0dd6771e278a10998396b027" (* Decode *)

open Owl
open Owl_types
open Neural 
open Neural.S


(* TODO *)
let load_prepared_imgs = None


let get_input_data img_name = 
  let _, _, _, img = LoadImage.img_to_owl img_name in
  let shape = Dense.Ndarray.S.shape img in
  let shape = Array.append [|1|] shape in
  let img = Dense.Ndarray.S.reshape img shape in 
  
  (* Preprocessing input image data to range [-1, 1] *)
  let img = Dense.Ndarray.S.div_scalar img 255. in 
  let img = Dense.Ndarray.S.sub_scalar img 0.5  in
  let img = Dense.Ndarray.S.mul_scalar img 2.   in
  img

let decode_predictions ?(top=5) preds = 
  let h = Decode.load_dict () in 
  let tp = Dense.Matrix.S.top preds top in 
  Printf.printf "\nTop %d Predictions:\n" top;
  Array.iteri (fun i x -> 
    Printf.printf "Prediction #%d (%.2f%%) : " i (preds.{x.(0), x.(1)} *. 100.);
    Printf.printf "%s\n" (Hashtbl.find h x.(1)) 
  ) tp

let _ = 
  let nn  = Graph.load "inception_owl.network" in 
  let img = get_input_data "panda1.ppm" in 
  let preds = Graph.model_cnn nn img in
  decode_predictions preds