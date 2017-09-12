#!/usr/bin/env owl

#zoo "9428a62a31dbea75511882ab8218076f" (* InceptionV3 *)

open Owl
open Neural.S
open Neural.S.Graph

(* START: extra functions *)

let make_vgg_network () =
  let nn = input [|224;224;3|]
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
    |> fully_connected ~act_typ:Activation.Softmax 1000
    |> get_network
  in 
  nn

let load_tiny_imagenet_vgg () = 
  Dense.Ndarray.S.load "tinyimagenet_224", 0, (Dense.Matrix.S.zeros 1 1)

let load_tiny_imagenet () = 
  Dense.Ndarray.S.load "tinyimagenet100", 0, (Dense.Matrix.S.zeros 1 1) (* 0s are just two random wildcard placeholders *)

let draw_samples_vgg x _ n = 
  assert (n < 100 && n > 0);
  let col_num = (Owl_dense_ndarray_generic.shape x).(0) in
  let a = Array.init col_num (fun i -> i) in
  let a = Owl_stats.choose a n |> Array.to_list in
  Owl_dense_ndarray.S.get_slice [L a; R []; R []; R []] x,
  (Owl_dense_matrix.S.zeros 1 1) (* a random wildcard placeholder *)

let draw_samples x _ n = 
  assert (n < 100 && n > 0);
  let col_num = (Owl_dense_ndarray_generic.shape x).(0) in
  let a = Array.init col_num (fun i -> i) in
  let a = Owl_stats.choose a n |> Array.to_list in
  Owl_dense_ndarray.S.get_slice [L a; R []; R []; R []] x,
  (Owl_dense_matrix.S.zeros 1 1) (* a random wildcard placeholder *)

(* ENDS: extra functions *)

(* repeat tests on each node *)
let num_test = 10 

let get_statistics nn_model load_data take_sample = 
  (* let network = Graph.load "inception_owl.network" in (* not available *) *) 
  let x, _, y = load_data () in
  let network = nn_model () in 
  Graph.init network;
  Graph. _remove_training_nodes network;
  let num_nodes = Array.length network.topo in 
  let re = Dense.Matrix.D.zeros num_test num_nodes in  

  for i = 0 to (num_test-1) do
      let x', _ = take_sample x y 1 in (* TinyImageNet.draw_samples x 1 in (* not available *) *)
      (* Graph.model_cnn network x' (* not available *)*)
      Graph.run (Algodiff.S.Arr x') network |> ignore; 
      let a = collect_meta network.topo in 
      let a = Dense.Matrix.D.of_arrays [|a|] in
      Dense.Matrix.D.copy_row_to a re i
  done;

  let re_avg = Dense.Matrix.D.average_rows re in 
  let re_std = Dense.Matrix.D.std ~axis:0 re in 

  let u = Dense.Matrix.D.(re_avg + re_std) in 
  let l = Dense.Matrix.D.(re_avg - re_std) in 
  num_nodes, re_avg, u, l


let get_output_size nn_model = 
  let nn = nn_model () in 
  let nodes = nn.topo in 
  let num_nodes = Array.length nodes in 
  let outs = Array.make num_nodes 0. in 

  Array.iteri (fun i n -> 
    let neu = n.neuron in 
    let out_shape = 
      match neu with 
      | Neuron.Conv2D a          -> a.out_shape
      | Neuron.Normalisation a   -> a.out_shape
      | Neuron.Activation a      -> a.out_shape
      | Neuron.MaxPool2D a       -> a.out_shape
      | Neuron.Concatenate a     -> a.out_shape
      | Neuron.AvgPool2D a       -> a.out_shape
      | Neuron.GlobalAvgPool2D a -> a.out_shape
      | Neuron.Linear a          -> a.out_shape
      | Neuron.FullyConnected a  -> a.out_shape
      | Neuron.Dropout a         -> a.out_shape
      | Neuron.Input a           -> a.out_shape
      | _ -> [|0;0;0|]
    in
    outs.(i) <- float_of_int (Array.fold_left ( * ) 1 out_shape)
  ) nodes;

  let out_sizes = Dense.Matrix.D.of_array outs 1 num_nodes in 
  let out_sizes = Dense.Matrix.D.div_scalar out_sizes (1024. *. 1024. /. 4.) in  (*float32 = 4B; use MB as unit *)
  out_sizes 

let get_nodes_name nn_model = 
  let nn = nn_model () in 
  let nodes = nn.topo in 
  let num_nodes = Array.length nodes in 
  let names = Array.make num_nodes "" in 
  Array.iteri (fun i n ->
    names.(i) <- n.name
  ) nodes;
  names


let filter_names ?(th=15) names sizes =  
  let len = Array.length names in 
  assert (len = Array.length sizes);
  let filtered_names = Array.make len "" in 
  Array.iteri (fun i y -> 
    if y < th then 
      filtered_names.(i) <- names.(i)
  ) sizes;
  filtered_names

(* 

let plot_latency_vgg ?(name="") = 
  let num_nodes, avg, u, l = get_statistics make_vgg_network Dataset.load_cifar_test_data Dataset.draw_samples_cifar in 
  let x = Dense.Matrix.D.sequential 1 num_nodes in 
  let h = Plot.create name in 
  Plot.(plot ~h ~spec:[ RGB (255,0,0); LineStyle 1; Marker "#[0x2299]"; MarkerSize 1. ] x avg);
  Plot.output h

*)

let plot_latency_vgg16 ?(name="") = 
  let threshold1 = 40. in  (* ms *) 

  (* let num_nodes, avg, u, l = get_statistics make_vgg_network load_tiny_imagenet_vgg draw_samples in 
  Dense.Matrix.D.save_txt avg "vgg16_avg.save.txt";*)

  let avg = Dense.Matrix.D.load_txt "vgg16_avg.save.txt" in 

  let output_sizes = get_output_size make_vgg_network in 
  let node_names   = get_nodes_name  make_vgg_network in 
  let num_nodes =  Array.length node_names in 

  (* let filtered_names = filter_names node_names output_sizes in *)
  let x = Dense.Matrix.D.sequential 1 num_nodes in 

  (*
  let x' = x |> Dense.Matrix.D.to_array |> Array.to_list in 
  let names = Array.to_list node_names in 
  let xticks = List.combine x' names in 
  *)

  let h = Plot.create ~m:1 ~n:2 name in 
  Plot.set_font_size h 2.;
  Plot.subplot h 0 0;
  Plot.set_title h "VGG Nodes Latency";
  Plot.set_ylabel h "Latency (ms)";
  Plot.set_xlabel h "Nodes";

  (* Plot.set_xticklabels h xticks; *)
  
  Plot.(bar ~h ~spec:[ RGB (28,28,28); LineStyle 1; Marker "#[0x2299]"] avg);
  Dense.Matrix.D.iteri (fun i j s -> 
    if s > threshold1 then
      Plot.(text ~h ~spec:[ RGB (0,255,0) ] ~dx:1.0 ~dy:(-3.) (float_of_int j) s node_names.(j) )
  ) avg; 
  
  Plot.subplot h 0 1;
  Plot.set_title h "VGG Nodes Output Size ";
  Plot.set_ylabel h "Output Size (MB)";
  Plot.set_xlabel h "Nodes";
  Plot.(bar ~h ~spec:[ RGB (28,28,28); LineStyle 1; Marker "#[0x0394]"; MarkerSize 1. ] output_sizes);
  Plot.output h

let _ = 
  (* plot_latency_inception ~name:"inception.pdf" *)
  plot_latency_vgg16 ~name:"vgg16.pdf"
  