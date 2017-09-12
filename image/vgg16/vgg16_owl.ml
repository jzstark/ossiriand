#!/usr/bin/env owl

open Owl
open Owl_types
open Algodiff.S

open Neural
open Neural.S
open Neural.S.Graph

let channel_last = true (* The same in Keras Conv layer *)
let include_top = true  (* if false, no final Dense layer *)
let img_size = 224      (* include_top = true means img_size have to be exact 224 *)
let classes = 1000

let obtain_input_shape () = None 

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
    |> flatten 
    |> fully_connected ~act_typ:Activation.Relu 4096
    |> fully_connected ~act_typ:Activation.Relu 4096
    |> fully_connected ~act_typ:Activation.Softmax classes 
    |> get_network
  in 
  print nn;
  nn