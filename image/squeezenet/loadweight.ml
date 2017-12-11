#!/usr/bin/env owl

#zoo "41380a6baa6c5a37931cd375d494eb57" (*SqueezeNet*)

open Owl
open Owl_types
open Bigarray
open Hdf5_caml

open Neural 
open Neural.S 
open Neural.S.Graph

let fname = "sqnet_owl.hdf5"
let input = H5.open_rdonly fname 

(* A helper function *)
let print_array a = 
  Printf.printf "[|";
  Array.iter (fun x -> Printf.printf "%i; " x) a ;
  Printf.printf "|]\n"

let _ = 

let h = Hashtbl.create 52 in 
for i = 0 to 25 do
  let nname = "conv2d_" ^ (string_of_int i) in 
  let w = H5.read_float_genarray input (nname ^ ":w")  C_layout in
  let b = H5.read_float_genarray input (nname ^ ":b")  C_layout in 
  
  Hashtbl.add h (nname ^ ":w") (Dense.Ndarray.Generic.cast_d2s w);
  Hashtbl.add h (nname ^ ":b") (Dense.Ndarray.Generic.cast_d2s b);
done;

let nn = Squeezenet.make_squeezenet 227 in 
Graph.init nn;
let nodes = nn.topo in

let count = ref 0 in
Array.iter (fun n -> 
  if (Neuron.to_name n.neuron) = "conv2d" then
    let wb = Neuron.mkpar n.neuron in 
    let nname = "conv2d_" ^ (string_of_int (! count)) in 
    Printf.printf "%s: " nname; 
    count := !count + 1;
    let w_new = Hashtbl.find h (nname ^ ":w") in
    let b_new = Hashtbl.find h (nname ^ ":b") in
    print_array (Dense.Ndarray.S.shape w_new);

    wb.(0) <- (Neuron.Arr w_new);
    wb.(1) <- (Neuron.Arr b_new);
    Neuron.update n.neuron wb
) nodes;

save nn "sqnet_owl.network"