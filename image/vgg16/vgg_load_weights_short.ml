#!/usr/bin/env owl

open Owl
open Owl_types
open Algodiff.S
open Neural 
open Neural.S 
open Neural.S.Graph
open Bigarray
open Hdf5_caml

let fname = "vgg16_owl.hdf5" 
let input_weights = H5.open_rdonly fname 

let img_size = 224
let classes = 1000

let model () = 
  let nn = input [|img_size;img_size;3|]
    (* block 1 *)
    |> conv2d [|3;3;3;64|] [|1;1|] ~act_typ:Activation.Relu ~padding:SAME 
    |> conv2d [|3;3;64;64|] [|1;1|] ~act_typ:Activation.Relu ~padding:SAME 
    |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
    (* block 2 *)
    |> conv2d [|3;3;64;128|] [|1;1|] ~act_typ:Activation.Relu ~padding:SAME 
    |> conv2d [|3;3;128;128|] [|1;1|] ~act_typ:Activation.Relu ~padding:SAME 
    |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
    (* block 3 *)
    |> conv2d [|3;3;128;256|] [|1;1|] ~act_typ:Activation.Relu ~padding:SAME 
    |> conv2d [|3;3;256;256|] [|1;1|] ~act_typ:Activation.Relu ~padding:SAME 
    |> conv2d [|3;3;256;256|] [|1;1|] ~act_typ:Activation.Relu ~padding:SAME
    |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
    (* block 4 *)
    |> conv2d [|3;3;256;512|] [|1;1|] ~act_typ:Activation.Relu ~padding:SAME 
    |> conv2d [|3;3;512;512|] [|1;1|] ~act_typ:Activation.Relu ~padding:SAME 
    |> conv2d [|3;3;512;512|] [|1;1|] ~act_typ:Activation.Relu ~padding:SAME
    |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
    (* block 5 *)
    |> conv2d [|3;3;512;512|] [|1;1|] ~act_typ:Activation.Relu ~padding:SAME 
    |> conv2d [|3;3;512;512|] [|1;1|] ~act_typ:Activation.Relu ~padding:SAME 
    |> conv2d [|3;3;512;512|] [|1;1|] ~act_typ:Activation.Relu ~padding:SAME
    |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
    (* classification block *)
    (*
    |> flatten 
    |> fully_connected ~act_typ:Activation.Relu 4096
    |> fully_connected ~act_typ:Activation.Relu 4096
    |> fully_connected ~act_typ:Activation.Softmax classes *)
    |> get_network
  in
  nn

(* A simple helper function *)
let print_array a = 
  Printf.printf "[|";
  Array.iter (fun x -> Printf.printf "%i; " x) a ;
  Printf.printf "|]\n"

let _ = 

let h = Hashtbl.create 26 in 
(* manually get; 13 conv2ds x 2, 2 fc x 2, and 1 pred x 2 *)
for i = 1 to 13 do
  let nname = "conv2d_" ^ (string_of_int i) in 
  let w = H5.read_float_genarray input_weights (nname ^ "_w") C_layout in 
  Hashtbl.add h (nname ^ "_w") (Dense.Ndarray.Generic.cast_d2s w);

  let b = H5.read_float_genarray input_weights (nname ^ "_b") C_layout in 
  Hashtbl.add h (nname ^ "_b") (Dense.Ndarray.Generic.cast_d2s b)
done;


let nn = model () in 
Graph.init nn;
let nodes = nn.topo in 

(* 1: fill all the conv nodes *)
let count = ref 1 in

Array.iter (fun (n : node) -> 
  if (Neuron.to_name n.neuron) = "conv2d" then
    let wb = Neuron.mkpar n.neuron in 
    let nname = "conv2d_" ^ (string_of_int (! count)) in 
    Printf.printf "%s: " nname; 
    count := !count + 1;
    let w_new = Hashtbl.find h (nname ^ "_w") in
    let b_new = Hashtbl.find h (nname ^ "_b") in
    print_array (Dense.Ndarray.S.shape w_new);

    wb.(0) <- (Neuron.Arr w_new);
    wb.(1) <- (Neuron.Arr b_new);
    Neuron.update n.neuron wb
) nodes ;

(* save the model *)
save nn "vgg16_owl_short.network"
