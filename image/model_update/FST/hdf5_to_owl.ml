open Owl
open Neural.S
open Neural.S.Graph
open Algodiff
open Hdf5_caml
open Bigarray
module N = Dense.Ndarray.S

let conv2d_n = "conv2d"
let conv_trans2d_n = "transpose_conv2d"
let norm_n = "normalisation"
let relu_n = "activation"

let conv2d_layer ?(relu=true) kernel stride nn  =
  let result = 
    conv2d ~padding:SAME kernel stride nn
    |> normalisation ~decay:0. ~training:true ~axis:3
  in
  match relu with
  | true -> (result |> activation Activation.Relu)
  | _    -> result

let conv2d_trans_layer kernel stride nn = 
  transpose_conv2d ~padding:SAME kernel stride nn
  |> normalisation ~decay:0. ~training:true ~axis:3
  |> activation Activation.Relu

let residual_block wh nn = 
  let tmp = conv2d_layer [|wh; wh; 128; 128|] [|1;1|] nn
    |> conv2d_layer ~relu:false [|wh; wh; 128; 128|] [|1;1|]
  in 
  add [|nn; tmp|]

let img_size = 256

(* Perfectly balanced -- like everything should be. *)
let make_network () = 
  input [|img_size;img_size;3|]
  |> conv2d_layer [|9;9;3;32|] [|1;1|]
  |> conv2d_layer [|3;3;32;64|] [|2;2|]
  |> conv2d_layer [|3;3;64;128|] [|2;2|]
  |> residual_block 3
  |> residual_block 3
  |> residual_block 3
  |> residual_block 3
  |> residual_block 3
  |> conv2d_trans_layer [|3;3;128;64|] [|2;2|]
  |> conv2d_trans_layer [|3;3;64;32|] [|2;2|]
  |> conv2d_layer ~relu:false [|9;9;32;3|] [|1;1|]
  |> lambda (fun x -> Maths.((tanh x) * (F 256.) + (F 127.5)))
  |> get_network


let layers = [|
  "conv2d_1";
  "normalisation_2_beta";
  "normalisation_2_gamma";
  "conv2d_4";
  "normalisation_5_beta";
  "normalisation_5_gamma";
  "conv2d_7";
  "normalisation_8_beta";
  "normalisation_8_gamma";
  "conv2d_10";
  "normalisation_11_beta";
  "normalisation_11_gamma";
  "conv2d_13";
  "normalisation_14_beta";
  "normalisation_14_gamma";
  "conv2d_16";
  "normalisation_17_beta";
  "normalisation_17_gamma";
  "conv2d_19";
  "normalisation_20_beta";
  "normalisation_20_gamma";
  "conv2d_22";
  "normalisation_23_beta";
  "normalisation_23_gamma";
  "conv2d_25";
  "normalisation_26_beta";
  "normalisation_26_gamma";
  "conv2d_28";
  "normalisation_29_beta";
  "normalisation_29_gamma";
  "conv2d_31";
  "normalisation_32_beta";
  "normalisation_32_gamma";
  "conv2d_34";
  "normalisation_35_beta";
  "normalisation_35_gamma";
  "conv2d_37";
  "normalisation_38_beta";
  "normalisation_38_gamma";
  "transpose_conv2d_40";
  "normalisation_41_beta";
  "normalisation_41_gamma";
  "transpose_conv2d_43";
  "normalisation_44_beta";
  "normalisation_44_gamma";
  "conv2d_46";
  "normalisation_47_beta";
  "normalisation_47_gamma"|]


(* First, call this function to change hdf5 to htb *)
let h5_to_htb h5file htbfile = 
  let h = Hashtbl.create 50 in
  let f = H5.open_rdonly h5file  in
  for i = 0 to (Array.length layers - 1) do
    let w = H5.read_float_genarray f layers.(i) C_layout in
    Hashtbl.add h layers.(i) (Dense.Ndarray.Generic.cast_d2s w)
  done;
  Owl_io.marshal_to_file h htbfile


(* Then load the weights from Hashtbl *)
let htb_to_weight htbfile weightfile = 

  let nn = make_network () in
  Graph.init nn;
  let nodes = nn.topo in 
  let h = Owl_io.marshal_from_file htbfile in

  Array.iter (fun n ->
    let nname = Neuron.to_name n.neuron in
    if nname = conv2d_n then (
      let wb = Neuron.mkpar n.neuron in
      Printf.printf "%s\n" n.name; 
      wb.(0) <- Algodiff.Arr (Hashtbl.find h n.name);
      Neuron.update n.neuron wb
    ) else if nname = norm_n then (
      Printf.printf "%s\n" n.name; 
      let be_ga = Neuron.mkpar n.neuron in
      be_ga.(0) <- Algodiff.Arr (Hashtbl.find h (n.name ^ "_beta"));
      be_ga.(1) <- Algodiff.Arr (Hashtbl.find h (n.name ^ "_gamma"));
      Neuron.update n.neuron be_ga
    ) else if nname = conv_trans2d_n then (
      Printf.printf "%s\n" n.name; 
      let wb = Neuron.mkpar n.neuron in
      let new_w = Hashtbl.find h n.name in
      let s = N.shape new_w in 
      let tmp = N.reshape new_w [|s.(0); s.(1); s.(3); s.(2)|] in
      (* let tmp = N.transpose ~axis:[|0;1;3;2|] new_w in 
      let s = N.shape tmp in
      for i = 0 to (s.(3) - 1) do
        for j = 0 to (s.(2) - 1) do
          let foo = N.get_slice [[];[];[j];[i]] tmp in
          let foo = Dense.Matrix.S.rotate foo 180 in
          let foo = N.reshape foo [|s.(0); s.(1); 1; 1|] in
          N.set_slice [[];[];[j];[i]] tmp foo;
        done; 
      done;*)
      wb.(0) <- Algodiff.Arr tmp;
      Neuron.update n.neuron wb
    )
  ) nodes;

  Graph.save_weights nn weightfile
  

let _ = 
  Array.iter (fun style_name -> 
    let hd5 = "fst_style_" ^ style_name ^ ".hdf5" in
    let htb = "fst_style_" ^ style_name ^ ".htb" in
    let owl = "fst_" ^ style_name ^ ".weight" in
    h5_to_htb hd5 htb; 
    htb_to_weight htb owl
  ) [|"la_muse"; "rain_princess"; "scream";"udnie";"wave";"wreck"|]
