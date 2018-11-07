open Owl
open Owl_types
open Bigarray
open Hdf5_caml

module N = Dense.Ndarray.S
module CPU_Engine = Owl_computation_cpu_engine.Make (N)
module CGCompiler = Owl_neural_compiler.Make (CPU_Engine)

open CGCompiler.Neural
open CGCompiler.Neural.Graph

let pack x = CGCompiler.Engine.pack_arr x |> Algodiff.pack_arr
let unpack x = Algodiff.unpack_arr x |> CGCompiler.Engine.unpack_arr

let fname = "sqnet_owl.hdf5"
let input = H5.open_rdonly fname 

let fire_module in_shape squeeze expand nn =
  let root = conv2d ~padding:VALID [|1;1; in_shape; squeeze|] [|1;1|] nn
    |> activation Activation.Relu in
  let left = conv2d ~padding:VALID [|1;1; squeeze;  expand |] [|1;1|] root
    |> activation Activation.Relu in
  let right = conv2d ~padding:SAME [|3;3; squeeze;  expand |] [|1;1|] root
    |> activation Activation.Relu in
  concatenate 3 [|left; right|]

let make_network img_size =
  Graph.input [|img_size;img_size;3|]
  |> conv2d ~padding:VALID [|3;3;3;64|] [|2;2|]
  |> max_pool2d [|3;3|] [|2;2|] ~padding:VALID
  (* block 1 *)
  |> fire_module 64  16 64
  |> fire_module 128 16 64
  |> max_pool2d [|3;3|] [|2;2|] ~padding:VALID
  (* block 2 *)
  |> fire_module 128 32 128
  |> fire_module 256 32 128
  |> max_pool2d [|3;3|] [|2;2|] ~padding:VALID
  (* block 3 *)
  |> fire_module 256 48 192
  |> fire_module 384 48 192
  |> fire_module 384 64 256
  |> fire_module 512 64 256
  (* include top *)
  |> dropout 0.5
  |> conv2d ~padding:VALID [|1;1;512;1000|] [|1;1|]
  |> activation Activation.Relu
  |> global_avg_pool2d
  |> activation Activation.(Softmax 1)
  |> get_network

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

let nn = make_network 227 in 
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
    print_array (N.shape w_new);

    wb.(0) <- pack w_new;
    wb.(1) <- pack b_new;
    Neuron.update n.neuron wb
) nodes;

save_weights nn "squeezenet_owl_cg.weight"

(* Bernoulli node could leads to slightly different results. *)