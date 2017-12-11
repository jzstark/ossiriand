#!/usr/bin/env owl

open Owl
open Neural.S
open Neural.S.Graph

let fire_module in_shape squeeze expand nn = 
  let root = conv2d ~padding:VALID [|1;1; in_shape; squeeze|] [|1;1|] nn
    |> activation Activation.Relu in
  let left = conv2d ~padding:VALID [|1;1; squeeze; expand |] [|1;1|] root
    |> activation Activation.Relu in
  let right = conv2d ~padding:SAME [|3;3; squeeze; expand |] [|1;1|] root
    |> activation Activation.Relu in
  concatenate 3 [|left; right|]


let make_squeezenet img_size =
  input [|img_size;img_size;3|]
  |> conv2d ~padding:VALID [|3;3;3;64|] [|2;2|]
  |> max_pool2d [|3;3|] [|2;2|] ~padding:VALID 

  |> fire_module 64  16 64
  |> fire_module 128 16 64
  |> max_pool2d [|3;3|] [|2;2|] ~padding:VALID 

  |> fire_module 128 32 128
  |> fire_module 256 32 128
  |> max_pool2d [|3;3|] [|2;2|] ~padding:VALID 

  |> fire_module 256 48 192
  |> fire_module 384 48 192
  |> fire_module 384 64 256
  |> fire_module 512 64 256

  (* include top *)
  |> dropout 0.5
  |> conv2d ~padding:VALID [|1;1;512;1000|] [|1;1|]
  |> activation Activation.Relu
  |> global_avg_pool2d
  |> activation Activation.Softmax
  |> get_network

let _ = make_squeezenet 227 |> print