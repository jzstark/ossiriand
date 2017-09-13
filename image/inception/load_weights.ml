#!/usr/bin/env owl

#zoo "9428a62a31dbea75511882ab8218076f" (* InceptionV3 *)

open Owl
open Owl_types
open Bigarray
open Hdf5_caml

open Neural 
open Neural.S 
open Neural.S.Graph

let fname = "incpetion_owl.hdf5" 
let input = H5.open_rdonly fname 

(* A simple helper function *)
let print_array a = 
  Printf.printf "[|";
  Array.iter (fun x -> Printf.printf "%i; " x) a ;
  (* Printf.printf "|]; Length: %d.\n" (a.(1) * a.(2) * a.(3)) *)
  Printf.printf "|]\n"


let _ = 

let h = Hashtbl.create 368 in (* manually get; 94 conv2ds, 94 batch norms x 3, and 2 for linear (w and b) *)
for i = 1 to 94 do
  let nname = "conv2d_" ^ (string_of_int i) in 
  let ws = H5.read_float_genarray input nname C_layout in 
  Hashtbl.add h nname (Dense.Ndarray.Generic.cast_d2s ws)
done;

for i = 1 to 94 do
  let nname = "batch_normalization_" ^ (string_of_int i) in 
  let beta = H5.read_float_genarray input (nname ^ "_beta")  C_layout in 
  let mean = H5.read_float_genarray input (nname ^ "_mean")  C_layout in 
  let var  = H5.read_float_genarray input (nname ^ "_var")   C_layout in 
  Hashtbl.add h (nname ^ "_beta") (Dense.Ndarray.Generic.cast_d2s beta);
  Hashtbl.add h (nname ^ "_mean") (Dense.Ndarray.Generic.cast_d2s mean);
  Hashtbl.add h (nname ^ "_var")  (Dense.Ndarray.Generic.cast_d2s var)
done;

let w = H5.read_float_genarray input "linear_w" C_layout in 
let b = H5.read_float_genarray input "linear_b" C_layout in 
Hashtbl.add h "linear_w" (Dense.Ndarray.Generic.cast_d2s w);
Hashtbl.add h "linear_b" (Dense.Ndarray.Generic.cast_d2s b); 
(*actually, we should not put the w and b as two seperate items, since they belong to a same node *)

(* load network structure *)
let nn = InceptionV3.model () in 

(* Graph.init nn *)
(* Array.iter (fun n -> Neuron.init n.neuron) nn.topo; *)
Graph.init nn;

let nodes = nn.topo in 

(* 1: fill all the conv nodes *)
let count = ref 1 in
(* Get weights sequentially *)
Array.iter (fun (n : node) -> 
  if (Neuron.to_name n.neuron) = "conv2d" then
    let wb = Neuron.mkpar n.neuron in 
    let nname = "conv2d_" ^ (string_of_int (! count)) in 
    Printf.printf "%s: " nname; 
    count := !count + 1;
    let w_new = Hashtbl.find h nname in
    print_array (Dense.Ndarray.S.shape w_new);

    wb.(0) <- (Neuron.Arr w_new);
    Neuron.update n.neuron wb
) nodes ;


(* 2:fill in all bn nodes *)

let count = ref 1 in
(* Get weights sequentially *)
Array.iter (fun (n : node) -> 
  if (Neuron.to_name n.neuron) = "normalisation" then

    let be_ga = Neuron.mkpar n.neuron in (*only beta, gamma*)
    let nname = "batch_normalization_" ^ (string_of_int (! count)) in 
    Printf.printf "%s: " nname; 
    count := !count + 1;
    let beta_new = Hashtbl.find h (nname ^ "_beta") in
    let mean_new = Hashtbl.find h (nname ^ "_mean") in 
    let var_new  = Hashtbl.find h (nname ^ "_var")  in 
    (* resize beta from a vector of len to ndarray of shape [|1;1;1;len|] <- shape hardcoded *)
    let len = Dense.Ndarray.S.shape beta_new in 

    let beta_new = Dense.Ndarray.S.reshape beta_new [|1;1;1;len.(0)|] in 
    let mean_new = Dense.Ndarray.S.reshape mean_new [|1;1;1;len.(0)|] in 
    let var_new  = Dense.Ndarray.S.reshape var_new  [|1;1;1;len.(0)|] in

    be_ga.(0) <- (Neuron.Arr beta_new);
    print_array (Dense.Ndarray.S.shape beta_new);
    (* be_ga.(2) <- (Neuron.Arr mean_new);
    be_ga.(3) <- (Neuron.Arr var_new); *)
    Neuron.update n.neuron be_ga;

    (* decouple from monad *)
    (function Neuron.Normalisation a -> (a.mu <- (Neuron.Arr mean_new))) n.neuron;
    (function Neuron.Normalisation a -> (a.var <- (Neuron.Arr var_new))) n.neuron
) nodes;

(* 3 : fill the final linear node *)

(* again, hard-coded to get the final linear node; 
   218(?) when there is node in conv2d_bn *)
let dense_node = nodes.(312) in 
let wb = Neuron.mkpar dense_node.neuron in 
(* get 2darry w and change to matrix *)
let w_new = Hashtbl.find h "linear_w" in 
let w_new = Dense.Matrix.S.of_ndarray w_new in
(* get 1darray b, reshape, and convert to matrix *)
let b_new = Hashtbl.find h "linear_b" in 
let b_dim = Array.append [|1|] (Dense.Ndarray.S.shape b_new) in 
let b_new = Dense.Ndarray.S.reshape b_new b_dim  in
let b_new = Dense.Matrix.S.of_ndarray b_new in 
(* fill in *)
wb.(0) <- (Neuron.Mat w_new);
wb.(1) <- (Neuron.Mat b_new);
Neuron.update dense_node.neuron wb;

(* save the model *)
save nn "inception_owl2.network"
