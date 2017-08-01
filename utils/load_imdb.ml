let vocab_file = "imdb.vocab"

let load_file f =
  let ic = open_in f in
  let n = in_channel_length ic in
  let s = Bytes.create n in
  really_input ic s 0 n;
  close_in ic;
  (* remove non alphabetic character except for whitespace and hyphen *)
  (* currently cannot process multiple hyphen input *)
  let s = Str.global_replace (Str.regexp "[^a-zA-Z- ']") "" s in 
  s |> String.lowercase_ascii

let string2array s = 
  Str.split (Str.regexp " ") s
  |> Array.of_list

(* Attrib: https://stackoverflow.com/questions/5774934/how-do-i-read-in-lines-from-a-text-file-in-ocaml*)
let read_lines name : string list =
  let ic = open_in name in
  let try_read () =
    try Some (input_line ic) with End_of_file -> None in
  let rec loop acc = match try_read () with
    | Some s -> loop (s :: acc)
    | None -> close_in ic; List.rev acc in
  loop []

let load_vocab vocab_f = 
  let lines = read_lines vocab_f in 
  let len = List.length lines  in 
  let v2i = Hashtbl.create len in 
  let i2v = Hashtbl.create len in 
  List.iteri (fun i v -> 
    Hashtbl.add v2i v i;
    Hashtbl.add i2v i v;
  ) lines;
  v2i, i2v


let vectorize_file fname = 
  let s = load_file fname in 
  let s = string2array s in 
  let num_str = Array.make (Array.length s) 0  in 
  let v2i, _ = load_vocab vocab_file in 
  Array.iteri (fun i w ->
    let index = 
    try 
      Hashtbl.find v2i w
    with 
      Not_found -> 0
    in 
    Array.set num_str i index 
  ) s; 
  num_str (* problem: zero index *)


let load_file ?(sign="pos") dir = 
  (* dir is of form './train' *)
  let dir = dir ^ "/" ^ sign in 
  let rec loop_files xs acc = 
    match xs with
    | []   -> acc 
    | h::t -> loop_files t (List.append acc [vectorize_file h]) 
  in 
  let file_list = Sys.readdir dir 
    |> Array.to_list  
    |> List.map  (fun f -> dir ^ "/" ^ f) 
  in 
  let x = loop_files file_list [] in 
  let y = match sign with
    | "pos" -> Array.make (List.length x) 1
    | _     -> Array.make (List.length x) 0
    (* add error for other input than "pos/neg" *)
  in
  x, y

(* need to combine pos and neg  *)
let x_train_pos, y_train_pos = load_file "./train"
let x_train_neg, y_train_neg = load_file ~sign:"neg" "./train"
let x_test_pos, y_test_pos = load_file "./test"
let x_test_neg, y_test_neg = load_file ~sign:"neg" "./test"
