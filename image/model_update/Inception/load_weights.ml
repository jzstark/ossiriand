open Owl
open Owl_types
open Bigarray
open Hdf5_caml

open Neural.S 
open Neural.S.Graph

let fname = "incpetion_owl.hdf5" 
let input = H5.open_rdonly fname 

let channel_last = true (* The same in Keras Conv layer *)
let include_top = true  (* if false, no final Dense layer *)
let img_size = 299      (* include_top = true means img_size have to be exact 299 *)

let conv2d_bn ?(padding=SAME) kernel stride nn =
  conv2d ~padding kernel stride nn
  |> normalisation ~training:false ~axis:3
  |> activation Activation.Relu

let mix_typ1 in_shape bp_size nn =
  let branch1x1 = conv2d_bn [|1;1;in_shape;64|] [|1;1|] nn in
  let branch5x5 = nn
    |> conv2d_bn [|1;1;in_shape;48|] [|1;1|]
    |> conv2d_bn [|5;5;48;64|] [|1;1|]
  in
  let branch3x3dbl = nn
    |> conv2d_bn [|1;1;in_shape;64|] [|1;1|]
    |> conv2d_bn [|3;3;64;96|]  [|1;1|]
    |> conv2d_bn [|3;3;96;96|]  [|1;1|]
  in
  let branch_pool = nn
    |> avg_pool2d [|3;3|] [|1;1|]
    |> conv2d_bn [|1;1;in_shape; bp_size |] [|1;1|]
  in
  concatenate 3 [|branch1x1; branch5x5; branch3x3dbl; branch_pool|]

let mix_typ3 nn =
  let branch3x3 = conv2d_bn [|3;3;288;384|] [|2;2|] ~padding:VALID nn in
  let branch3x3dbl = nn
    |> conv2d_bn [|1;1;288;64|] [|1;1|]
    |> conv2d_bn [|3;3;64;96|] [|1;1|]
    |> conv2d_bn [|3;3;96;96|] [|2;2|] ~padding:VALID
  in
  let branch_pool = max_pool2d [|3;3|] [|2;2|] ~padding:VALID nn in
  concatenate 3 [|branch3x3; branch3x3dbl; branch_pool|]

let mix_typ4 size nn =
  let branch1x1 = conv2d_bn [|1;1;768;192|] [|1;1|] nn in
  let branch7x7 = nn
    |> conv2d_bn [|1;1;768;size|] [|1;1|]
    |> conv2d_bn [|1;7;size;size|] [|1;1|]
    |> conv2d_bn [|7;1;size;192|] [|1;1|]
  in
  let branch7x7dbl = nn
    |> conv2d_bn [|1;1;768;size|] [|1;1|]
    |> conv2d_bn [|7;1;size;size|] [|1;1|]
    |> conv2d_bn [|1;7;size;size|] [|1;1|]
    |> conv2d_bn [|7;1;size;size|] [|1;1|]
    |> conv2d_bn [|1;7;size;192|] [|1;1|]
  in
  let branch_pool = nn
    |> avg_pool2d [|3;3|] [|1;1|] (* padding = SAME *)
    |> conv2d_bn [|1;1; 768; 192|] [|1;1|]
  in
  concatenate 3 [|branch1x1; branch7x7; branch7x7dbl; branch_pool|]

let mix_typ8 nn =
  let branch3x3 = nn
    |> conv2d_bn [|1;1;768;192|] [|1;1|]
    |> conv2d_bn [|3;3;192;320|] [|2;2|] ~padding:VALID
  in
  let branch7x7x3 = nn
    |> conv2d_bn [|1;1;768;192|] [|1;1|]
    |> conv2d_bn [|1;7;192;192|] [|1;1|]
    |> conv2d_bn [|7;1;192;192|] [|1;1|]
    |> conv2d_bn [|3;3;192;192|] [|2;2|] ~padding:VALID
  in
  let branch_pool = max_pool2d [|3;3|] [|2;2|] ~padding:VALID nn in
  concatenate 3 [|branch3x3; branch7x7x3; branch_pool|]

let mix_typ9 input nn =
  let branch1x1 = conv2d_bn [|1;1;input;320|] [|1;1|] nn in
  let branch3x3 = conv2d_bn [|1;1;input;384|] [|1;1|] nn in
  let branch3x3_1 = branch3x3 |> conv2d_bn [|1;3;384;384|] [|1;1|] in
  let branch3x3_2 = branch3x3 |> conv2d_bn [|3;1;384;384|] [|1;1|] in
  let branch3x3 = concatenate 3 [| branch3x3_1; branch3x3_2 |] in
  let branch3x3dbl = nn |> conv2d_bn [|1;1;input;448|] [|1;1|] |> conv2d_bn [|3;3;448;384|] [|1;1|] in
  let branch3x3dbl_1 = branch3x3dbl |> conv2d_bn [|1;3;384;384|] [|1;1|]  in
  let branch3x3dbl_2 = branch3x3dbl |> conv2d_bn [|3;1;384;384|] [|1;1|]  in
  let branch3x3dbl = concatenate 3 [|branch3x3dbl_1; branch3x3dbl_2|] in
  let branch_pool = nn |> avg_pool2d [|3;3|] [|1;1|] |> conv2d_bn [|1;1;input;192|] [|1;1|] in
  concatenate 3 [|branch1x1; branch3x3; branch3x3dbl; branch_pool|]

let make_network img_size =
  Graph.input [|img_size;img_size;3|]
  |> conv2d_bn [|3;3;3;32|] [|2;2|] ~padding:VALID
  |> conv2d_bn [|3;3;32;32|] [|1;1|] ~padding:VALID
  |> conv2d_bn [|3;3;32;64|] [|1;1|]
  |> max_pool2d [|3;3|] [|2;2|] ~padding:VALID
  |> conv2d_bn [|1;1;64;80|] [|1;1|] ~padding:VALID
  |> conv2d_bn [|3;3;80;192|] [|1;1|] ~padding:VALID
  |> max_pool2d [|3;3|] [|2;2|] ~padding:VALID
  |> mix_typ1 192 32 |> mix_typ1 256 64 |> mix_typ1 288 64
  |> mix_typ3
  |> mix_typ4 128 |> mix_typ4 160 |> mix_typ4 160 |> mix_typ4 192
  |> mix_typ8
  |> mix_typ9 1280 |> mix_typ9 2048
  |> global_avg_pool2d
  |> linear 1000 ~act_typ:Activation.(Softmax 1)
  |> get_network


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
(* actually, we should not put the w and b as two seperate items, since they belong to a same node *)

(* run this step if you want to save the hash table first *)
(* Owl_utils.marshal_to_file h "inceptionv3.htb";
let h = Owl_io.marshal_from_file "inceptionv3.htb"; *)


(* load network structure *)
let nn = make_network 299 in 

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

    wb.(0) <- (Algodiff.Arr w_new);
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

    be_ga.(0) <- (Algodiff.Arr beta_new);
    print_array (Dense.Ndarray.S.shape beta_new);
    (* be_ga.(2) <- (Algodiff.Arr mean_new);
    be_ga.(3) <- (Algodiff.Arr var_new); *)
    Neuron.update n.neuron be_ga;

    (* decouple from monad *)
    (function Neuron.Normalisation a -> (a.mu <- (Algodiff.Arr mean_new))) n.neuron;
    (function Neuron.Normalisation a -> (a.var <- (Algodiff.Arr var_new))) n.neuron
) nodes;

(* 3 : fill the final linear node *)

(* again, hard-coded to get the final linear node; 
   218(?) when there is node in conv2d_bn *)
let dense_node = nodes.(312) in 
let wb = Neuron.mkpar dense_node.neuron in 
(* get 2darry w and change to matrix *)
let w_new = Hashtbl.find h "linear_w" in 
(* get 1darray b, reshape, and convert to matrix *)
let b_new = Hashtbl.find h "linear_b" in 
let b_dim = Array.append [|1|] (Dense.Ndarray.S.shape b_new) in 
let b_new = Dense.Ndarray.S.reshape b_new b_dim  in
(* fill in *)
wb.(0) <- (Algodiff.Arr w_new);
wb.(1) <- (Algodiff.Arr b_new);
Neuron.update dense_node.neuron wb;

(* save the model *)
save_weights nn "inception_owl.weight"
