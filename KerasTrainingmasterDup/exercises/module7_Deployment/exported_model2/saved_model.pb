э
ъ
9
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
p
	AssignAdd
ref"T

value"T

output_ref"T"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
:
Minimum
x"T
y"T
z"T"
Ttype:	
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
D
NotEqual
x"T
y"T
z
"
Ttype:
2	

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
5
Pow
x"T
y"T
z"T"
Ttype:
	2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
,
Sqrt
x"T
y"T"
Ttype:	
2
0
Square
x"T
y"T"
Ttype:
	2	
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.3.02v1.3.0-rc2-20-g0787eeeпЉ
p
layer_1_inputPlaceholder*
dtype0*
shape:џџџџџџџџџ	*'
_output_shapes
:џџџџџџџџџ	
m
layer_1/random_uniform/shapeConst*
valueB"	   2   *
dtype0*
_output_shapes
:
_
layer_1/random_uniform/minConst*
valueB
 *dFЃО*
dtype0*
_output_shapes
: 
_
layer_1/random_uniform/maxConst*
valueB
 *dFЃ>*
dtype0*
_output_shapes
: 
Ј
$layer_1/random_uniform/RandomUniformRandomUniformlayer_1/random_uniform/shape*
seedБџх)*
seed2цнс*
dtype0*
T0*
_output_shapes

:	2
z
layer_1/random_uniform/subSublayer_1/random_uniform/maxlayer_1/random_uniform/min*
T0*
_output_shapes
: 

layer_1/random_uniform/mulMul$layer_1/random_uniform/RandomUniformlayer_1/random_uniform/sub*
T0*
_output_shapes

:	2
~
layer_1/random_uniformAddlayer_1/random_uniform/mullayer_1/random_uniform/min*
T0*
_output_shapes

:	2

layer_1/kernel
VariableV2*
shape
:	2*
dtype0*
	container *
shared_name *
_output_shapes

:	2
М
layer_1/kernel/AssignAssignlayer_1/kernellayer_1/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@layer_1/kernel*
_output_shapes

:	2
{
layer_1/kernel/readIdentitylayer_1/kernel*
T0*!
_class
loc:@layer_1/kernel*
_output_shapes

:	2
Z
layer_1/ConstConst*
valueB2*    *
dtype0*
_output_shapes
:2
x
layer_1/bias
VariableV2*
shape:2*
dtype0*
	container *
shared_name *
_output_shapes
:2
Љ
layer_1/bias/AssignAssignlayer_1/biaslayer_1/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@layer_1/bias*
_output_shapes
:2
q
layer_1/bias/readIdentitylayer_1/bias*
T0*
_class
loc:@layer_1/bias*
_output_shapes
:2

layer_1/MatMulMatMullayer_1_inputlayer_1/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ2

layer_1/BiasAddBiasAddlayer_1/MatMullayer_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ2
W
layer_1/ReluRelulayer_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ2
m
layer_2/random_uniform/shapeConst*
valueB"2   d   *
dtype0*
_output_shapes
:
_
layer_2/random_uniform/minConst*
valueB
 *ЭЬLО*
dtype0*
_output_shapes
: 
_
layer_2/random_uniform/maxConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Ј
$layer_2/random_uniform/RandomUniformRandomUniformlayer_2/random_uniform/shape*
seedБџх)*
seed2њЯЄ*
dtype0*
T0*
_output_shapes

:2d
z
layer_2/random_uniform/subSublayer_2/random_uniform/maxlayer_2/random_uniform/min*
T0*
_output_shapes
: 

layer_2/random_uniform/mulMul$layer_2/random_uniform/RandomUniformlayer_2/random_uniform/sub*
T0*
_output_shapes

:2d
~
layer_2/random_uniformAddlayer_2/random_uniform/mullayer_2/random_uniform/min*
T0*
_output_shapes

:2d

layer_2/kernel
VariableV2*
shape
:2d*
dtype0*
	container *
shared_name *
_output_shapes

:2d
М
layer_2/kernel/AssignAssignlayer_2/kernellayer_2/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@layer_2/kernel*
_output_shapes

:2d
{
layer_2/kernel/readIdentitylayer_2/kernel*
T0*!
_class
loc:@layer_2/kernel*
_output_shapes

:2d
Z
layer_2/ConstConst*
valueBd*    *
dtype0*
_output_shapes
:d
x
layer_2/bias
VariableV2*
shape:d*
dtype0*
	container *
shared_name *
_output_shapes
:d
Љ
layer_2/bias/AssignAssignlayer_2/biaslayer_2/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@layer_2/bias*
_output_shapes
:d
q
layer_2/bias/readIdentitylayer_2/bias*
T0*
_class
loc:@layer_2/bias*
_output_shapes
:d

layer_2/MatMulMatMullayer_1/Relulayer_2/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџd

layer_2/BiasAddBiasAddlayer_2/MatMullayer_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџd
W
layer_2/ReluRelulayer_2/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџd
m
layer_3/random_uniform/shapeConst*
valueB"d   2   *
dtype0*
_output_shapes
:
_
layer_3/random_uniform/minConst*
valueB
 *ЭЬLО*
dtype0*
_output_shapes
: 
_
layer_3/random_uniform/maxConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Ј
$layer_3/random_uniform/RandomUniformRandomUniformlayer_3/random_uniform/shape*
seedБџх)*
seed2А*
dtype0*
T0*
_output_shapes

:d2
z
layer_3/random_uniform/subSublayer_3/random_uniform/maxlayer_3/random_uniform/min*
T0*
_output_shapes
: 

layer_3/random_uniform/mulMul$layer_3/random_uniform/RandomUniformlayer_3/random_uniform/sub*
T0*
_output_shapes

:d2
~
layer_3/random_uniformAddlayer_3/random_uniform/mullayer_3/random_uniform/min*
T0*
_output_shapes

:d2

layer_3/kernel
VariableV2*
shape
:d2*
dtype0*
	container *
shared_name *
_output_shapes

:d2
М
layer_3/kernel/AssignAssignlayer_3/kernellayer_3/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@layer_3/kernel*
_output_shapes

:d2
{
layer_3/kernel/readIdentitylayer_3/kernel*
T0*!
_class
loc:@layer_3/kernel*
_output_shapes

:d2
Z
layer_3/ConstConst*
valueB2*    *
dtype0*
_output_shapes
:2
x
layer_3/bias
VariableV2*
shape:2*
dtype0*
	container *
shared_name *
_output_shapes
:2
Љ
layer_3/bias/AssignAssignlayer_3/biaslayer_3/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@layer_3/bias*
_output_shapes
:2
q
layer_3/bias/readIdentitylayer_3/bias*
T0*
_class
loc:@layer_3/bias*
_output_shapes
:2

layer_3/MatMulMatMullayer_2/Relulayer_3/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ2

layer_3/BiasAddBiasAddlayer_3/MatMullayer_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ2
W
layer_3/ReluRelulayer_3/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ2
r
!output_layer/random_uniform/shapeConst*
valueB"2      *
dtype0*
_output_shapes
:
d
output_layer/random_uniform/minConst*
valueB
 *SЏО*
dtype0*
_output_shapes
: 
d
output_layer/random_uniform/maxConst*
valueB
 *SЏ>*
dtype0*
_output_shapes
: 
Б
)output_layer/random_uniform/RandomUniformRandomUniform!output_layer/random_uniform/shape*
seedБџх)*
seed2дБ6*
dtype0*
T0*
_output_shapes

:2

output_layer/random_uniform/subSuboutput_layer/random_uniform/maxoutput_layer/random_uniform/min*
T0*
_output_shapes
: 

output_layer/random_uniform/mulMul)output_layer/random_uniform/RandomUniformoutput_layer/random_uniform/sub*
T0*
_output_shapes

:2

output_layer/random_uniformAddoutput_layer/random_uniform/muloutput_layer/random_uniform/min*
T0*
_output_shapes

:2

output_layer/kernel
VariableV2*
shape
:2*
dtype0*
	container *
shared_name *
_output_shapes

:2
а
output_layer/kernel/AssignAssignoutput_layer/kerneloutput_layer/random_uniform*
T0*
validate_shape(*
use_locking(*&
_class
loc:@output_layer/kernel*
_output_shapes

:2

output_layer/kernel/readIdentityoutput_layer/kernel*
T0*&
_class
loc:@output_layer/kernel*
_output_shapes

:2
_
output_layer/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
}
output_layer/bias
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
Н
output_layer/bias/AssignAssignoutput_layer/biasoutput_layer/Const*
T0*
validate_shape(*
use_locking(*$
_class
loc:@output_layer/bias*
_output_shapes
:

output_layer/bias/readIdentityoutput_layer/bias*
T0*$
_class
loc:@output_layer/bias*
_output_shapes
:

output_layer/MatMulMatMullayer_3/Reluoutput_layer/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ

output_layer/BiasAddBiasAddoutput_layer/MatMuloutput_layer/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
b
Adam/iterations/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
s
Adam/iterations
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
О
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
T0*
validate_shape(*
use_locking(*"
_class
loc:@Adam/iterations*
_output_shapes
: 
v
Adam/iterations/readIdentityAdam/iterations*
T0*"
_class
loc:@Adam/iterations*
_output_shapes
: 
Z
Adam/lr/initial_valueConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 

Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
Adam/beta_1
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
Ў
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@Adam/beta_1*
_output_shapes
: 
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
valueB
 *wО?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
Ў
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@Adam/beta_2*
_output_shapes
: 
j
Adam/beta_2/readIdentityAdam/beta_2*
T0*
_class
loc:@Adam/beta_2*
_output_shapes
: 
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
Њ
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@Adam/decay*
_output_shapes
: 
g
Adam/decay/readIdentity
Adam/decay*
T0*
_class
loc:@Adam/decay*
_output_shapes
: 

output_layer_targetPlaceholder*
dtype0*%
shape:џџџџџџџџџџџџџџџџџџ*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
v
output_layer_sample_weightsPlaceholder*
dtype0*
shape:џџџџџџџџџ*#
_output_shapes
:џџџџџџџџџ

loss/output_layer_loss/subSuboutput_layer/BiasAddoutput_layer_target*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
~
loss/output_layer_loss/SquareSquareloss/output_layer_loss/sub*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
x
-loss/output_layer_loss/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
М
loss/output_layer_loss/MeanMeanloss/output_layer_loss/Square-loss/output_layer_loss/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
r
/loss/output_layer_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
О
loss/output_layer_loss/Mean_1Meanloss/output_layer_loss/Mean/loss/output_layer_loss/Mean_1/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ

loss/output_layer_loss/mulMulloss/output_layer_loss/Mean_1output_layer_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
f
!loss/output_layer_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/output_layer_loss/NotEqualNotEqualoutput_layer_sample_weights!loss/output_layer_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ

loss/output_layer_loss/CastCastloss/output_layer_loss/NotEqual*

SrcT0
*

DstT0*#
_output_shapes
:џџџџџџџџџ
f
loss/output_layer_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/output_layer_loss/Mean_2Meanloss/output_layer_loss/Castloss/output_layer_loss/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 

loss/output_layer_loss/truedivRealDivloss/output_layer_loss/mulloss/output_layer_loss/Mean_2*
T0*#
_output_shapes
:џџџџџџџџџ
h
loss/output_layer_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ѓ
loss/output_layer_loss/Mean_3Meanloss/output_layer_loss/truedivloss/output_layer_loss/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
[
loss/mulMul
loss/mul/xloss/output_layer_loss/Mean_3*
T0*
_output_shapes
: 
}
training/Adam/gradients/ShapeConst*
valueB *
dtype0*
_class
loc:@loss/mul*
_output_shapes
: 

training/Adam/gradients/ConstConst*
valueB
 *  ?*
dtype0*
_class
loc:@loss/mul*
_output_shapes
: 
 
training/Adam/gradients/FillFilltraining/Adam/gradients/Shapetraining/Adam/gradients/Const*
T0*
_class
loc:@loss/mul*
_output_shapes
: 

+training/Adam/gradients/loss/mul_grad/ShapeConst*
valueB *
dtype0*
_class
loc:@loss/mul*
_output_shapes
: 

-training/Adam/gradients/loss/mul_grad/Shape_1Const*
valueB *
dtype0*
_class
loc:@loss/mul*
_output_shapes
: 

;training/Adam/gradients/loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs+training/Adam/gradients/loss/mul_grad/Shape-training/Adam/gradients/loss/mul_grad/Shape_1*
T0*
_class
loc:@loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
)training/Adam/gradients/loss/mul_grad/mulMultraining/Adam/gradients/Fillloss/output_layer_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
ѕ
)training/Adam/gradients/loss/mul_grad/SumSum)training/Adam/gradients/loss/mul_grad/mul;training/Adam/gradients/loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_class
loc:@loss/mul*
_output_shapes
:
м
-training/Adam/gradients/loss/mul_grad/ReshapeReshape)training/Adam/gradients/loss/mul_grad/Sum+training/Adam/gradients/loss/mul_grad/Shape*
T0*
Tshape0*
_class
loc:@loss/mul*
_output_shapes
: 

+training/Adam/gradients/loss/mul_grad/mul_1Mul
loss/mul/xtraining/Adam/gradients/Fill*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
ћ
+training/Adam/gradients/loss/mul_grad/Sum_1Sum+training/Adam/gradients/loss/mul_grad/mul_1=training/Adam/gradients/loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_class
loc:@loss/mul*
_output_shapes
:
т
/training/Adam/gradients/loss/mul_grad/Reshape_1Reshape+training/Adam/gradients/loss/mul_grad/Sum_1-training/Adam/gradients/loss/mul_grad/Shape_1*
T0*
Tshape0*
_class
loc:@loss/mul*
_output_shapes
: 
Ф
Htraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Reshape/shapeConst*
valueB:*
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*
_output_shapes
:
­
Btraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/ReshapeReshape/training/Adam/gradients/loss/mul_grad/Reshape_1Htraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*
_output_shapes
:
а
@training/Adam/gradients/loss/output_layer_loss/Mean_3_grad/ShapeShapeloss/output_layer_loss/truediv*
T0*
out_type0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*
_output_shapes
:
П
?training/Adam/gradients/loss/output_layer_loss/Mean_3_grad/TileTileBtraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Reshape@training/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Shape*
T0*

Tmultiples0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
в
Btraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Shape_1Shapeloss/output_layer_loss/truediv*
T0*
out_type0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*
_output_shapes
:
З
Btraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Shape_2Const*
valueB *
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*
_output_shapes
: 
М
@training/Adam/gradients/loss/output_layer_loss/Mean_3_grad/ConstConst*
valueB: *
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*
_output_shapes
:
Н
?training/Adam/gradients/loss/output_layer_loss/Mean_3_grad/ProdProdBtraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Shape_1@training/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Const*
	keep_dims( *
T0*

Tidx0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*
_output_shapes
: 
О
Btraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Const_1Const*
valueB: *
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*
_output_shapes
:
С
Atraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Prod_1ProdBtraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Shape_2Btraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Const_1*
	keep_dims( *
T0*

Tidx0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*
_output_shapes
: 
И
Dtraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*
_output_shapes
: 
Љ
Btraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/MaximumMaximumAtraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Prod_1Dtraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Maximum/y*
T0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*
_output_shapes
: 
Ї
Ctraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/floordivFloorDiv?training/Adam/gradients/loss/output_layer_loss/Mean_3_grad/ProdBtraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Maximum*
T0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*
_output_shapes
: 
ю
?training/Adam/gradients/loss/output_layer_loss/Mean_3_grad/CastCastCtraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/floordiv*

SrcT0*

DstT0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*
_output_shapes
: 
Џ
Btraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/truedivRealDiv?training/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Tile?training/Adam/gradients/loss/output_layer_loss/Mean_3_grad/Cast*
T0*0
_class&
$"loc:@loss/output_layer_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Ю
Atraining/Adam/gradients/loss/output_layer_loss/truediv_grad/ShapeShapeloss/output_layer_loss/mul*
T0*
out_type0*1
_class'
%#loc:@loss/output_layer_loss/truediv*
_output_shapes
:
Й
Ctraining/Adam/gradients/loss/output_layer_loss/truediv_grad/Shape_1Const*
valueB *
dtype0*1
_class'
%#loc:@loss/output_layer_loss/truediv*
_output_shapes
: 
т
Qtraining/Adam/gradients/loss/output_layer_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/Adam/gradients/loss/output_layer_loss/truediv_grad/ShapeCtraining/Adam/gradients/loss/output_layer_loss/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@loss/output_layer_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ctraining/Adam/gradients/loss/output_layer_loss/truediv_grad/RealDivRealDivBtraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/truedivloss/output_layer_loss/Mean_2*
T0*1
_class'
%#loc:@loss/output_layer_loss/truediv*#
_output_shapes
:џџџџџџџџџ
б
?training/Adam/gradients/loss/output_layer_loss/truediv_grad/SumSumCtraining/Adam/gradients/loss/output_layer_loss/truediv_grad/RealDivQtraining/Adam/gradients/loss/output_layer_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*1
_class'
%#loc:@loss/output_layer_loss/truediv*
_output_shapes
:
С
Ctraining/Adam/gradients/loss/output_layer_loss/truediv_grad/ReshapeReshape?training/Adam/gradients/loss/output_layer_loss/truediv_grad/SumAtraining/Adam/gradients/loss/output_layer_loss/truediv_grad/Shape*
T0*
Tshape0*1
_class'
%#loc:@loss/output_layer_loss/truediv*#
_output_shapes
:џџџџџџџџџ
У
?training/Adam/gradients/loss/output_layer_loss/truediv_grad/NegNegloss/output_layer_loss/mul*
T0*1
_class'
%#loc:@loss/output_layer_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Etraining/Adam/gradients/loss/output_layer_loss/truediv_grad/RealDiv_1RealDiv?training/Adam/gradients/loss/output_layer_loss/truediv_grad/Negloss/output_layer_loss/Mean_2*
T0*1
_class'
%#loc:@loss/output_layer_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Etraining/Adam/gradients/loss/output_layer_loss/truediv_grad/RealDiv_2RealDivEtraining/Adam/gradients/loss/output_layer_loss/truediv_grad/RealDiv_1loss/output_layer_loss/Mean_2*
T0*1
_class'
%#loc:@loss/output_layer_loss/truediv*#
_output_shapes
:џџџџџџџџџ
В
?training/Adam/gradients/loss/output_layer_loss/truediv_grad/mulMulBtraining/Adam/gradients/loss/output_layer_loss/Mean_3_grad/truedivEtraining/Adam/gradients/loss/output_layer_loss/truediv_grad/RealDiv_2*
T0*1
_class'
%#loc:@loss/output_layer_loss/truediv*#
_output_shapes
:џџџџџџџџџ
б
Atraining/Adam/gradients/loss/output_layer_loss/truediv_grad/Sum_1Sum?training/Adam/gradients/loss/output_layer_loss/truediv_grad/mulStraining/Adam/gradients/loss/output_layer_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*1
_class'
%#loc:@loss/output_layer_loss/truediv*
_output_shapes
:
К
Etraining/Adam/gradients/loss/output_layer_loss/truediv_grad/Reshape_1ReshapeAtraining/Adam/gradients/loss/output_layer_loss/truediv_grad/Sum_1Ctraining/Adam/gradients/loss/output_layer_loss/truediv_grad/Shape_1*
T0*
Tshape0*1
_class'
%#loc:@loss/output_layer_loss/truediv*
_output_shapes
: 
Щ
=training/Adam/gradients/loss/output_layer_loss/mul_grad/ShapeShapeloss/output_layer_loss/Mean_1*
T0*
out_type0*-
_class#
!loc:@loss/output_layer_loss/mul*
_output_shapes
:
Щ
?training/Adam/gradients/loss/output_layer_loss/mul_grad/Shape_1Shapeoutput_layer_sample_weights*
T0*
out_type0*-
_class#
!loc:@loss/output_layer_loss/mul*
_output_shapes
:
в
Mtraining/Adam/gradients/loss/output_layer_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/loss/output_layer_loss/mul_grad/Shape?training/Adam/gradients/loss/output_layer_loss/mul_grad/Shape_1*
T0*-
_class#
!loc:@loss/output_layer_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

;training/Adam/gradients/loss/output_layer_loss/mul_grad/mulMulCtraining/Adam/gradients/loss/output_layer_loss/truediv_grad/Reshapeoutput_layer_sample_weights*
T0*-
_class#
!loc:@loss/output_layer_loss/mul*#
_output_shapes
:џџџџџџџџџ
Н
;training/Adam/gradients/loss/output_layer_loss/mul_grad/SumSum;training/Adam/gradients/loss/output_layer_loss/mul_grad/mulMtraining/Adam/gradients/loss/output_layer_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*-
_class#
!loc:@loss/output_layer_loss/mul*
_output_shapes
:
Б
?training/Adam/gradients/loss/output_layer_loss/mul_grad/ReshapeReshape;training/Adam/gradients/loss/output_layer_loss/mul_grad/Sum=training/Adam/gradients/loss/output_layer_loss/mul_grad/Shape*
T0*
Tshape0*-
_class#
!loc:@loss/output_layer_loss/mul*#
_output_shapes
:џџџџџџџџџ

=training/Adam/gradients/loss/output_layer_loss/mul_grad/mul_1Mulloss/output_layer_loss/Mean_1Ctraining/Adam/gradients/loss/output_layer_loss/truediv_grad/Reshape*
T0*-
_class#
!loc:@loss/output_layer_loss/mul*#
_output_shapes
:џџџџџџџџџ
У
=training/Adam/gradients/loss/output_layer_loss/mul_grad/Sum_1Sum=training/Adam/gradients/loss/output_layer_loss/mul_grad/mul_1Otraining/Adam/gradients/loss/output_layer_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*-
_class#
!loc:@loss/output_layer_loss/mul*
_output_shapes
:
З
Atraining/Adam/gradients/loss/output_layer_loss/mul_grad/Reshape_1Reshape=training/Adam/gradients/loss/output_layer_loss/mul_grad/Sum_1?training/Adam/gradients/loss/output_layer_loss/mul_grad/Shape_1*
T0*
Tshape0*-
_class#
!loc:@loss/output_layer_loss/mul*#
_output_shapes
:џџџџџџџџџ
Э
@training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/ShapeShapeloss/output_layer_loss/Mean*
T0*
out_type0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
:
Г
?training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/SizeConst*
value	B :*
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 

>training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/addAdd/loss/output_layer_loss/Mean_1/reduction_indices?training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Size*
T0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 
 
>training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/modFloorMod>training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/add?training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Size*
T0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 
О
Btraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Shape_1Const*
valueB: *
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
:
К
Ftraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/range/startConst*
value	B : *
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 
К
Ftraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/range/deltaConst*
value	B :*
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 
є
@training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/rangeRangeFtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/range/start?training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/SizeFtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/range/delta*

Tidx0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
:
Й
Etraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Fill/valueConst*
value	B :*
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 
Ї
?training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/FillFillBtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Shape_1Etraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Fill/value*
T0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 
Ч
Htraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/DynamicStitchDynamicStitch@training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/range>training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/mod@training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Shape?training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Fill*
N*
T0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
И
Dtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 
Н
Btraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/MaximumMaximumHtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/DynamicStitchDtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Maximum/y*
T0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
Е
Ctraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/floordivFloorDiv@training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/ShapeBtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Maximum*
T0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
Л
Btraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/ReshapeReshape?training/Adam/gradients/loss/output_layer_loss/mul_grad/ReshapeHtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/DynamicStitch*
T0*
Tshape0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
:
З
?training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/TileTileBtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/ReshapeCtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/floordiv*
T0*

Tmultiples0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
:
Я
Btraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Shape_2Shapeloss/output_layer_loss/Mean*
T0*
out_type0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
:
б
Btraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Shape_3Shapeloss/output_layer_loss/Mean_1*
T0*
out_type0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
:
М
@training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/ConstConst*
valueB: *
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
:
Н
?training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/ProdProdBtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Shape_2@training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Const*
	keep_dims( *
T0*

Tidx0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 
О
Btraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Const_1Const*
valueB: *
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
:
С
Atraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Prod_1ProdBtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Shape_3Btraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Const_1*
	keep_dims( *
T0*

Tidx0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 
К
Ftraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Maximum_1/yConst*
value	B :*
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 
­
Dtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Maximum_1MaximumAtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Prod_1Ftraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Maximum_1/y*
T0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 
Ћ
Etraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/floordiv_1FloorDiv?training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/ProdDtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Maximum_1*
T0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 
№
?training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/CastCastEtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/floordiv_1*

SrcT0*

DstT0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*
_output_shapes
: 
Џ
Btraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/truedivRealDiv?training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Tile?training/Adam/gradients/loss/output_layer_loss/Mean_1_grad/Cast*
T0*0
_class&
$"loc:@loss/output_layer_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
Ы
>training/Adam/gradients/loss/output_layer_loss/Mean_grad/ShapeShapeloss/output_layer_loss/Square*
T0*
out_type0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
:
Џ
=training/Adam/gradients/loss/output_layer_loss/Mean_grad/SizeConst*
value	B :*
dtype0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 

<training/Adam/gradients/loss/output_layer_loss/Mean_grad/addAdd-loss/output_layer_loss/Mean/reduction_indices=training/Adam/gradients/loss/output_layer_loss/Mean_grad/Size*
T0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 

<training/Adam/gradients/loss/output_layer_loss/Mean_grad/modFloorMod<training/Adam/gradients/loss/output_layer_loss/Mean_grad/add=training/Adam/gradients/loss/output_layer_loss/Mean_grad/Size*
T0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 
Г
@training/Adam/gradients/loss/output_layer_loss/Mean_grad/Shape_1Const*
valueB *
dtype0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 
Ж
Dtraining/Adam/gradients/loss/output_layer_loss/Mean_grad/range/startConst*
value	B : *
dtype0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 
Ж
Dtraining/Adam/gradients/loss/output_layer_loss/Mean_grad/range/deltaConst*
value	B :*
dtype0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 
ъ
>training/Adam/gradients/loss/output_layer_loss/Mean_grad/rangeRangeDtraining/Adam/gradients/loss/output_layer_loss/Mean_grad/range/start=training/Adam/gradients/loss/output_layer_loss/Mean_grad/SizeDtraining/Adam/gradients/loss/output_layer_loss/Mean_grad/range/delta*

Tidx0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
:
Е
Ctraining/Adam/gradients/loss/output_layer_loss/Mean_grad/Fill/valueConst*
value	B :*
dtype0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 

=training/Adam/gradients/loss/output_layer_loss/Mean_grad/FillFill@training/Adam/gradients/loss/output_layer_loss/Mean_grad/Shape_1Ctraining/Adam/gradients/loss/output_layer_loss/Mean_grad/Fill/value*
T0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 
Л
Ftraining/Adam/gradients/loss/output_layer_loss/Mean_grad/DynamicStitchDynamicStitch>training/Adam/gradients/loss/output_layer_loss/Mean_grad/range<training/Adam/gradients/loss/output_layer_loss/Mean_grad/mod>training/Adam/gradients/loss/output_layer_loss/Mean_grad/Shape=training/Adam/gradients/loss/output_layer_loss/Mean_grad/Fill*
N*
T0*.
_class$
" loc:@loss/output_layer_loss/Mean*#
_output_shapes
:џџџџџџџџџ
Д
Btraining/Adam/gradients/loss/output_layer_loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 
Е
@training/Adam/gradients/loss/output_layer_loss/Mean_grad/MaximumMaximumFtraining/Adam/gradients/loss/output_layer_loss/Mean_grad/DynamicStitchBtraining/Adam/gradients/loss/output_layer_loss/Mean_grad/Maximum/y*
T0*.
_class$
" loc:@loss/output_layer_loss/Mean*#
_output_shapes
:џџџџџџџџџ
Є
Atraining/Adam/gradients/loss/output_layer_loss/Mean_grad/floordivFloorDiv>training/Adam/gradients/loss/output_layer_loss/Mean_grad/Shape@training/Adam/gradients/loss/output_layer_loss/Mean_grad/Maximum*
T0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
:
И
@training/Adam/gradients/loss/output_layer_loss/Mean_grad/ReshapeReshapeBtraining/Adam/gradients/loss/output_layer_loss/Mean_1_grad/truedivFtraining/Adam/gradients/loss/output_layer_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
:
Ч
=training/Adam/gradients/loss/output_layer_loss/Mean_grad/TileTile@training/Adam/gradients/loss/output_layer_loss/Mean_grad/ReshapeAtraining/Adam/gradients/loss/output_layer_loss/Mean_grad/floordiv*
T0*

Tmultiples0*.
_class$
" loc:@loss/output_layer_loss/Mean*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Э
@training/Adam/gradients/loss/output_layer_loss/Mean_grad/Shape_2Shapeloss/output_layer_loss/Square*
T0*
out_type0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
:
Ы
@training/Adam/gradients/loss/output_layer_loss/Mean_grad/Shape_3Shapeloss/output_layer_loss/Mean*
T0*
out_type0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
:
И
>training/Adam/gradients/loss/output_layer_loss/Mean_grad/ConstConst*
valueB: *
dtype0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
:
Е
=training/Adam/gradients/loss/output_layer_loss/Mean_grad/ProdProd@training/Adam/gradients/loss/output_layer_loss/Mean_grad/Shape_2>training/Adam/gradients/loss/output_layer_loss/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 
К
@training/Adam/gradients/loss/output_layer_loss/Mean_grad/Const_1Const*
valueB: *
dtype0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
:
Й
?training/Adam/gradients/loss/output_layer_loss/Mean_grad/Prod_1Prod@training/Adam/gradients/loss/output_layer_loss/Mean_grad/Shape_3@training/Adam/gradients/loss/output_layer_loss/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 
Ж
Dtraining/Adam/gradients/loss/output_layer_loss/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 
Ѕ
Btraining/Adam/gradients/loss/output_layer_loss/Mean_grad/Maximum_1Maximum?training/Adam/gradients/loss/output_layer_loss/Mean_grad/Prod_1Dtraining/Adam/gradients/loss/output_layer_loss/Mean_grad/Maximum_1/y*
T0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 
Ѓ
Ctraining/Adam/gradients/loss/output_layer_loss/Mean_grad/floordiv_1FloorDiv=training/Adam/gradients/loss/output_layer_loss/Mean_grad/ProdBtraining/Adam/gradients/loss/output_layer_loss/Mean_grad/Maximum_1*
T0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 
ъ
=training/Adam/gradients/loss/output_layer_loss/Mean_grad/CastCastCtraining/Adam/gradients/loss/output_layer_loss/Mean_grad/floordiv_1*

SrcT0*

DstT0*.
_class$
" loc:@loss/output_layer_loss/Mean*
_output_shapes
: 
Д
@training/Adam/gradients/loss/output_layer_loss/Mean_grad/truedivRealDiv=training/Adam/gradients/loss/output_layer_loss/Mean_grad/Tile=training/Adam/gradients/loss/output_layer_loss/Mean_grad/Cast*
T0*.
_class$
" loc:@loss/output_layer_loss/Mean*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
њ
@training/Adam/gradients/loss/output_layer_loss/Square_grad/mul/xConstA^training/Adam/gradients/loss/output_layer_loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*0
_class&
$"loc:@loss/output_layer_loss/Square*
_output_shapes
: 

>training/Adam/gradients/loss/output_layer_loss/Square_grad/mulMul@training/Adam/gradients/loss/output_layer_loss/Square_grad/mul/xloss/output_layer_loss/sub*
T0*0
_class&
$"loc:@loss/output_layer_loss/Square*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ж
@training/Adam/gradients/loss/output_layer_loss/Square_grad/mul_1Mul@training/Adam/gradients/loss/output_layer_loss/Mean_grad/truediv>training/Adam/gradients/loss/output_layer_loss/Square_grad/mul*
T0*0
_class&
$"loc:@loss/output_layer_loss/Square*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
=training/Adam/gradients/loss/output_layer_loss/sub_grad/ShapeShapeoutput_layer/BiasAdd*
T0*
out_type0*-
_class#
!loc:@loss/output_layer_loss/sub*
_output_shapes
:
С
?training/Adam/gradients/loss/output_layer_loss/sub_grad/Shape_1Shapeoutput_layer_target*
T0*
out_type0*-
_class#
!loc:@loss/output_layer_loss/sub*
_output_shapes
:
в
Mtraining/Adam/gradients/loss/output_layer_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs=training/Adam/gradients/loss/output_layer_loss/sub_grad/Shape?training/Adam/gradients/loss/output_layer_loss/sub_grad/Shape_1*
T0*-
_class#
!loc:@loss/output_layer_loss/sub*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Т
;training/Adam/gradients/loss/output_layer_loss/sub_grad/SumSum@training/Adam/gradients/loss/output_layer_loss/Square_grad/mul_1Mtraining/Adam/gradients/loss/output_layer_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*-
_class#
!loc:@loss/output_layer_loss/sub*
_output_shapes
:
Е
?training/Adam/gradients/loss/output_layer_loss/sub_grad/ReshapeReshape;training/Adam/gradients/loss/output_layer_loss/sub_grad/Sum=training/Adam/gradients/loss/output_layer_loss/sub_grad/Shape*
T0*
Tshape0*-
_class#
!loc:@loss/output_layer_loss/sub*'
_output_shapes
:џџџџџџџџџ
Ц
=training/Adam/gradients/loss/output_layer_loss/sub_grad/Sum_1Sum@training/Adam/gradients/loss/output_layer_loss/Square_grad/mul_1Otraining/Adam/gradients/loss/output_layer_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*-
_class#
!loc:@loss/output_layer_loss/sub*
_output_shapes
:
г
;training/Adam/gradients/loss/output_layer_loss/sub_grad/NegNeg=training/Adam/gradients/loss/output_layer_loss/sub_grad/Sum_1*
T0*-
_class#
!loc:@loss/output_layer_loss/sub*
_output_shapes
:
Т
Atraining/Adam/gradients/loss/output_layer_loss/sub_grad/Reshape_1Reshape;training/Adam/gradients/loss/output_layer_loss/sub_grad/Neg?training/Adam/gradients/loss/output_layer_loss/sub_grad/Shape_1*
T0*
Tshape0*-
_class#
!loc:@loss/output_layer_loss/sub*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ђ
=training/Adam/gradients/output_layer/BiasAdd_grad/BiasAddGradBiasAddGrad?training/Adam/gradients/loss/output_layer_loss/sub_grad/Reshape*
T0*
data_formatNHWC*'
_class
loc:@output_layer/BiasAdd*
_output_shapes
:

7training/Adam/gradients/output_layer/MatMul_grad/MatMulMatMul?training/Adam/gradients/loss/output_layer_loss/sub_grad/Reshapeoutput_layer/kernel/read*
transpose_a( *
transpose_b(*
T0*&
_class
loc:@output_layer/MatMul*'
_output_shapes
:џџџџџџџџџ2

9training/Adam/gradients/output_layer/MatMul_grad/MatMul_1MatMullayer_3/Relu?training/Adam/gradients/loss/output_layer_loss/sub_grad/Reshape*
transpose_a(*
transpose_b( *
T0*&
_class
loc:@output_layer/MatMul*
_output_shapes

:2
и
2training/Adam/gradients/layer_3/Relu_grad/ReluGradReluGrad7training/Adam/gradients/output_layer/MatMul_grad/MatMullayer_3/Relu*
T0*
_class
loc:@layer_3/Relu*'
_output_shapes
:џџџџџџџџџ2
л
8training/Adam/gradients/layer_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/layer_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@layer_3/BiasAdd*
_output_shapes
:2

2training/Adam/gradients/layer_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/layer_3/Relu_grad/ReluGradlayer_3/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@layer_3/MatMul*'
_output_shapes
:џџџџџџџџџd
ђ
4training/Adam/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu2training/Adam/gradients/layer_3/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@layer_3/MatMul*
_output_shapes

:d2
г
2training/Adam/gradients/layer_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/layer_3/MatMul_grad/MatMullayer_2/Relu*
T0*
_class
loc:@layer_2/Relu*'
_output_shapes
:џџџџџџџџџd
л
8training/Adam/gradients/layer_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/layer_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@layer_2/BiasAdd*
_output_shapes
:d

2training/Adam/gradients/layer_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/layer_2/Relu_grad/ReluGradlayer_2/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@layer_2/MatMul*'
_output_shapes
:џџџџџџџџџ2
ђ
4training/Adam/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu2training/Adam/gradients/layer_2/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@layer_2/MatMul*
_output_shapes

:2d
г
2training/Adam/gradients/layer_1/Relu_grad/ReluGradReluGrad2training/Adam/gradients/layer_2/MatMul_grad/MatMullayer_1/Relu*
T0*
_class
loc:@layer_1/Relu*'
_output_shapes
:џџџџџџџџџ2
л
8training/Adam/gradients/layer_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/layer_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@layer_1/BiasAdd*
_output_shapes
:2

2training/Adam/gradients/layer_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/layer_1/Relu_grad/ReluGradlayer_1/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@layer_1/MatMul*'
_output_shapes
:џџџџџџџџџ	
ѓ
4training/Adam/gradients/layer_1/MatMul_grad/MatMul_1MatMullayer_1_input2training/Adam/gradients/layer_1/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@layer_1/MatMul*
_output_shapes

:	2
b
training/Adam/AssignAdd/valueConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ќ
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
T0*
use_locking( *"
_class
loc:@Adam/iterations*
_output_shapes
: 
X
training/Adam/add/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/addAddAdam/iterations/readtraining/Adam/add/y*
T0*
_output_shapes
: 
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
T0*
_output_shapes
: 
X
training/Adam/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_1Const*
valueB
 *  *
dtype0*
_output_shapes
: 
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
T0*
_output_shapes
: 

training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
T0*
_output_shapes
: 
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
j
training/Adam/Const_2Const*
valueB	2*    *
dtype0*
_output_shapes

:	2

training/Adam/Variable
VariableV2*
shape
:	2*
dtype0*
	container *
shared_name *
_output_shapes

:	2
г
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/Const_2*
T0*
validate_shape(*
use_locking(*)
_class
loc:@training/Adam/Variable*
_output_shapes

:	2

training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*)
_class
loc:@training/Adam/Variable*
_output_shapes

:	2
b
training/Adam/Const_3Const*
valueB2*    *
dtype0*
_output_shapes
:2

training/Adam/Variable_1
VariableV2*
shape:2*
dtype0*
	container *
shared_name *
_output_shapes
:2
е
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/Const_3*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:2

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:2
j
training/Adam/Const_4Const*
valueB2d*    *
dtype0*
_output_shapes

:2d

training/Adam/Variable_2
VariableV2*
shape
:2d*
dtype0*
	container *
shared_name *
_output_shapes

:2d
й
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/Const_4*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes

:2d

training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
T0*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes

:2d
b
training/Adam/Const_5Const*
valueBd*    *
dtype0*
_output_shapes
:d

training/Adam/Variable_3
VariableV2*
shape:d*
dtype0*
	container *
shared_name *
_output_shapes
:d
е
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/Const_5*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:d

training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
T0*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:d
j
training/Adam/Const_6Const*
valueBd2*    *
dtype0*
_output_shapes

:d2

training/Adam/Variable_4
VariableV2*
shape
:d2*
dtype0*
	container *
shared_name *
_output_shapes

:d2
й
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/Const_6*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes

:d2

training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes

:d2
b
training/Adam/Const_7Const*
valueB2*    *
dtype0*
_output_shapes
:2

training/Adam/Variable_5
VariableV2*
shape:2*
dtype0*
	container *
shared_name *
_output_shapes
:2
е
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/Const_7*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
:2

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
:2
j
training/Adam/Const_8Const*
valueB2*    *
dtype0*
_output_shapes

:2

training/Adam/Variable_6
VariableV2*
shape
:2*
dtype0*
	container *
shared_name *
_output_shapes

:2
й
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/Const_8*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes

:2

training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes

:2
b
training/Adam/Const_9Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_7
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
е
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/Const_9*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:
k
training/Adam/Const_10Const*
valueB	2*    *
dtype0*
_output_shapes

:	2

training/Adam/Variable_8
VariableV2*
shape
:	2*
dtype0*
	container *
shared_name *
_output_shapes

:	2
к
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/Const_10*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes

:	2

training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes

:	2
c
training/Adam/Const_11Const*
valueB2*    *
dtype0*
_output_shapes
:2

training/Adam/Variable_9
VariableV2*
shape:2*
dtype0*
	container *
shared_name *
_output_shapes
:2
ж
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/Const_11*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:2

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:2
k
training/Adam/Const_12Const*
valueB2d*    *
dtype0*
_output_shapes

:2d

training/Adam/Variable_10
VariableV2*
shape
:2d*
dtype0*
	container *
shared_name *
_output_shapes

:2d
н
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/Const_12*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes

:2d

training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes

:2d
c
training/Adam/Const_13Const*
valueBd*    *
dtype0*
_output_shapes
:d

training/Adam/Variable_11
VariableV2*
shape:d*
dtype0*
	container *
shared_name *
_output_shapes
:d
й
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/Const_13*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:d

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:d
k
training/Adam/Const_14Const*
valueBd2*    *
dtype0*
_output_shapes

:d2

training/Adam/Variable_12
VariableV2*
shape
:d2*
dtype0*
	container *
shared_name *
_output_shapes

:d2
н
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/Const_14*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes

:d2

training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes

:d2
c
training/Adam/Const_15Const*
valueB2*    *
dtype0*
_output_shapes
:2

training/Adam/Variable_13
VariableV2*
shape:2*
dtype0*
	container *
shared_name *
_output_shapes
:2
й
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/Const_15*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
:2

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
T0*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
:2
k
training/Adam/Const_16Const*
valueB2*    *
dtype0*
_output_shapes

:2

training/Adam/Variable_14
VariableV2*
shape
:2*
dtype0*
	container *
shared_name *
_output_shapes

:2
н
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/Const_16*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes

:2

training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes

:2
c
training/Adam/Const_17Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_15
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
й
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/Const_17*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:
r
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
T0*
_output_shapes

:	2
Z
training/Adam/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_2Multraining/Adam/sub_24training/Adam/gradients/layer_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:	2
m
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*
_output_shapes

:	2
t
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
T0*
_output_shapes

:	2
Z
training/Adam/sub_3/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 
}
training/Adam/SquareSquare4training/Adam/gradients/layer_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:	2
n
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*
_output_shapes

:	2
m
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*
_output_shapes

:	2
k
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*
_output_shapes

:	2
[
training/Adam/Const_18Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_19Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_19*
T0*
_output_shapes

:	2

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_18*
T0*
_output_shapes

:	2
d
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*
_output_shapes

:	2
Z
training/Adam/add_3/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
p
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*
_output_shapes

:	2
u
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*
_output_shapes

:	2
q
training/Adam/sub_4Sublayer_1/kernel/readtraining/Adam/truediv_1*
T0*
_output_shapes

:	2
Ш
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
T0*
validate_shape(*
use_locking(*)
_class
loc:@training/Adam/Variable*
_output_shapes

:	2
Ю
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes

:	2
К
training/Adam/Assign_2Assignlayer_1/kerneltraining/Adam/sub_4*
T0*
validate_shape(*
use_locking(*!
_class
loc:@layer_1/kernel*
_output_shapes

:	2
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes
:2
Z
training/Adam/sub_5/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_7Multraining/Adam/sub_58training/Adam/gradients/layer_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:2
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:2
p
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
T0*
_output_shapes
:2
Z
training/Adam/sub_6/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_1Square8training/Adam/gradients/layer_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:2
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
:2
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:2
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
:2
[
training/Adam/Const_20Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_21Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_21*
T0*
_output_shapes
:2

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_20*
T0*
_output_shapes
:2
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
:2
Z
training/Adam/add_6/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes
:2
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
:2
k
training/Adam/sub_7Sublayer_1/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
:2
Ъ
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:2
Ъ
training/Adam/Assign_4Assigntraining/Adam/Variable_9training/Adam/add_5*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:2
В
training/Adam/Assign_5Assignlayer_1/biastraining/Adam/sub_7*
T0*
validate_shape(*
use_locking(*
_class
loc:@layer_1/bias*
_output_shapes
:2
u
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*
_output_shapes

:2d
Z
training/Adam/sub_8/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_12Multraining/Adam/sub_84training/Adam/gradients/layer_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:2d
o
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*
_output_shapes

:2d
v
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0*
_output_shapes

:2d
Z
training/Adam/sub_9/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_2Square4training/Adam/gradients/layer_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:2d
q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes

:2d
o
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*
_output_shapes

:2d
l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*
_output_shapes

:2d
[
training/Adam/Const_22Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_23Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_23*
T0*
_output_shapes

:2d

training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_22*
T0*
_output_shapes

:2d
d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes

:2d
Z
training/Adam/add_9/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
p
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*
_output_shapes

:2d
v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*
_output_shapes

:2d
r
training/Adam/sub_10Sublayer_2/kernel/readtraining/Adam/truediv_3*
T0*
_output_shapes

:2d
Ю
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes

:2d
а
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes

:2d
Л
training/Adam/Assign_8Assignlayer_2/kerneltraining/Adam/sub_10*
T0*
validate_shape(*
use_locking(*!
_class
loc:@layer_2/kernel*
_output_shapes

:2d
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes
:d
[
training/Adam/sub_11/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_17Multraining/Adam/sub_118training/Adam/gradients/layer_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:d
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:d
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
:d
[
training/Adam/sub_12/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_3Square8training/Adam/gradients/layer_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:d
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes
:d
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
:d
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
:d
[
training/Adam/Const_24Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_25Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_25*
T0*
_output_shapes
:d

training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_24*
T0*
_output_shapes
:d
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:d
[
training/Adam/add_12/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
:d
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes
:d
l
training/Adam/sub_13Sublayer_2/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes
:d
Ы
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:d
Ю
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:d
Д
training/Adam/Assign_11Assignlayer_2/biastraining/Adam/sub_13*
T0*
validate_shape(*
use_locking(*
_class
loc:@layer_2/bias*
_output_shapes
:d
u
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*
_output_shapes

:d2
[
training/Adam/sub_14/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/layer_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:d2
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes

:d2
v
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read*
T0*
_output_shapes

:d2
[
training/Adam/sub_15/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_4Square4training/Adam/gradients/layer_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:d2
r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*
_output_shapes

:d2
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*
_output_shapes

:d2
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*
_output_shapes

:d2
[
training/Adam/Const_26Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_27Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_27*
T0*
_output_shapes

:d2

training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_26*
T0*
_output_shapes

:d2
d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*
_output_shapes

:d2
[
training/Adam/add_15/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
r
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*
_output_shapes

:d2
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*
_output_shapes

:d2
r
training/Adam/sub_16Sublayer_3/kernel/readtraining/Adam/truediv_5*
T0*
_output_shapes

:d2
а
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes

:d2
в
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes

:d2
М
training/Adam/Assign_14Assignlayer_3/kerneltraining/Adam/sub_16*
T0*
validate_shape(*
use_locking(*!
_class
loc:@layer_3/kernel*
_output_shapes

:d2
q
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0*
_output_shapes
:2
[
training/Adam/sub_17/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_27Multraining/Adam/sub_178training/Adam/gradients/layer_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:2
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes
:2
r
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_13/read*
T0*
_output_shapes
:2
[
training/Adam/sub_18/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_5Square8training/Adam/gradients/layer_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:2
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes
:2
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes
:2
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes
:2
[
training/Adam/Const_28Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_29Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_29*
T0*
_output_shapes
:2

training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_28*
T0*
_output_shapes
:2
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes
:2
[
training/Adam/add_18/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
n
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
T0*
_output_shapes
:2
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes
:2
l
training/Adam/sub_19Sublayer_3/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes
:2
Ь
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
:2
Ю
training/Adam/Assign_16Assigntraining/Adam/Variable_13training/Adam/add_17*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
:2
Д
training/Adam/Assign_17Assignlayer_3/biastraining/Adam/sub_19*
T0*
validate_shape(*
use_locking(*
_class
loc:@layer_3/bias*
_output_shapes
:2
u
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes

:2
[
training/Adam/sub_20/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_32Multraining/Adam/sub_209training/Adam/gradients/output_layer/MatMul_grad/MatMul_1*
T0*
_output_shapes

:2
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes

:2
v
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*
_output_shapes

:2
[
training/Adam/sub_21/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_6Square9training/Adam/gradients/output_layer/MatMul_grad/MatMul_1*
T0*
_output_shapes

:2
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes

:2
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes

:2
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes

:2
[
training/Adam/Const_30Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_31Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_31*
T0*
_output_shapes

:2

training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_30*
T0*
_output_shapes

:2
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*
_output_shapes

:2
[
training/Adam/add_21/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
r
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*
_output_shapes

:2
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes

:2
w
training/Adam/sub_22Suboutput_layer/kernel/readtraining/Adam/truediv_7*
T0*
_output_shapes

:2
а
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes

:2
в
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes

:2
Ц
training/Adam/Assign_20Assignoutput_layer/kerneltraining/Adam/sub_22*
T0*
validate_shape(*
use_locking(*&
_class
loc:@output_layer/kernel*
_output_shapes

:2
q
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
T0*
_output_shapes
:
[
training/Adam/sub_23/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_37Multraining/Adam/sub_23=training/Adam/gradients/output_layer/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
T0*
_output_shapes
:
r
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_15/read*
T0*
_output_shapes
:
[
training/Adam/sub_24/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_7Square=training/Adam/gradients/output_layer/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes
:
l
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes
:
i
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
T0*
_output_shapes
:
[
training/Adam/Const_32Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_33Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_33*
T0*
_output_shapes
:

training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_32*
T0*
_output_shapes
:
`
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes
:
[
training/Adam/add_24/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
n
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
T0*
_output_shapes
:
s
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes
:
q
training/Adam/sub_25Suboutput_layer/bias/readtraining/Adam/truediv_8*
T0*
_output_shapes
:
Ь
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:
Ю
training/Adam/Assign_22Assigntraining/Adam/Variable_15training/Adam/add_23*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:
О
training/Adam/Assign_23Assignoutput_layer/biastraining/Adam/sub_25*
T0*
validate_shape(*
use_locking(*$
_class
loc:@output_layer/bias*
_output_shapes
:
Є
training/group_depsNoOp	^loss/mul^training/Adam/AssignAdd^training/Adam/Assign^training/Adam/Assign_1^training/Adam/Assign_2^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23
к
initNoOp^layer_1/kernel/Assign^layer_1/bias/Assign^layer_2/kernel/Assign^layer_2/bias/Assign^layer_3/kernel/Assign^layer_3/bias/Assign^output_layer/kernel/Assign^output_layer/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign ^training/Adam/Variable_2/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign


group_depsNoOp	^loss/mul
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_0b7f24bbeab54e3090c4b33c329dece8/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Ш
save/SaveV2/tensor_namesConst*ћ
valueёBюBAdam/beta_1BAdam/beta_2B
Adam/decayBAdam/iterationsBAdam/lrBlayer_1/biasBlayer_1/kernelBlayer_2/biasBlayer_2/kernelBlayer_3/biasBlayer_3/kernelBoutput_layer/biasBoutput_layer/kernelBtraining/Adam/VariableBtraining/Adam/Variable_1Btraining/Adam/Variable_10Btraining/Adam/Variable_11Btraining/Adam/Variable_12Btraining/Adam/Variable_13Btraining/Adam/Variable_14Btraining/Adam/Variable_15Btraining/Adam/Variable_2Btraining/Adam/Variable_3Btraining/Adam/Variable_4Btraining/Adam/Variable_5Btraining/Adam/Variable_6Btraining/Adam/Variable_7Btraining/Adam/Variable_8Btraining/Adam/Variable_9*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
і
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesAdam/beta_1Adam/beta_2
Adam/decayAdam/iterationsAdam/lrlayer_1/biaslayer_1/kernellayer_2/biaslayer_2/kernellayer_3/biaslayer_3/kerneloutput_layer/biasoutput_layer/kerneltraining/Adam/Variabletraining/Adam/Variable_1training/Adam/Variable_10training/Adam/Variable_11training/Adam/Variable_12training/Adam/Variable_13training/Adam/Variable_14training/Adam/Variable_15training/Adam/Variable_2training/Adam/Variable_3training/Adam/Variable_4training/Adam/Variable_5training/Adam/Variable_6training/Adam/Variable_7training/Adam/Variable_8training/Adam/Variable_9*+
dtypes!
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
T0*

axis *
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
o
save/RestoreV2/tensor_namesConst* 
valueBBAdam/beta_1*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:

save/AssignAssignAdam/beta_1save/RestoreV2*
T0*
validate_shape(*
use_locking(*
_class
loc:@Adam/beta_1*
_output_shapes
: 
q
save/RestoreV2_1/tensor_namesConst* 
valueBBAdam/beta_2*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
 
save/Assign_1AssignAdam/beta_2save/RestoreV2_1*
T0*
validate_shape(*
use_locking(*
_class
loc:@Adam/beta_2*
_output_shapes
: 
p
save/RestoreV2_2/tensor_namesConst*
valueBB
Adam/decay*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:

save/Assign_2Assign
Adam/decaysave/RestoreV2_2*
T0*
validate_shape(*
use_locking(*
_class
loc:@Adam/decay*
_output_shapes
: 
u
save/RestoreV2_3/tensor_namesConst*$
valueBBAdam/iterations*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
Ј
save/Assign_3AssignAdam/iterationssave/RestoreV2_3*
T0*
validate_shape(*
use_locking(*"
_class
loc:@Adam/iterations*
_output_shapes
: 
m
save/RestoreV2_4/tensor_namesConst*
valueBBAdam/lr*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:

save/Assign_4AssignAdam/lrsave/RestoreV2_4*
T0*
validate_shape(*
use_locking(*
_class
loc:@Adam/lr*
_output_shapes
: 
r
save/RestoreV2_5/tensor_namesConst*!
valueBBlayer_1/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
І
save/Assign_5Assignlayer_1/biassave/RestoreV2_5*
T0*
validate_shape(*
use_locking(*
_class
loc:@layer_1/bias*
_output_shapes
:2
t
save/RestoreV2_6/tensor_namesConst*#
valueBBlayer_1/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ў
save/Assign_6Assignlayer_1/kernelsave/RestoreV2_6*
T0*
validate_shape(*
use_locking(*!
_class
loc:@layer_1/kernel*
_output_shapes

:	2
r
save/RestoreV2_7/tensor_namesConst*!
valueBBlayer_2/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
І
save/Assign_7Assignlayer_2/biassave/RestoreV2_7*
T0*
validate_shape(*
use_locking(*
_class
loc:@layer_2/bias*
_output_shapes
:d
t
save/RestoreV2_8/tensor_namesConst*#
valueBBlayer_2/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ў
save/Assign_8Assignlayer_2/kernelsave/RestoreV2_8*
T0*
validate_shape(*
use_locking(*!
_class
loc:@layer_2/kernel*
_output_shapes

:2d
r
save/RestoreV2_9/tensor_namesConst*!
valueBBlayer_3/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
І
save/Assign_9Assignlayer_3/biassave/RestoreV2_9*
T0*
validate_shape(*
use_locking(*
_class
loc:@layer_3/bias*
_output_shapes
:2
u
save/RestoreV2_10/tensor_namesConst*#
valueBBlayer_3/kernel*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
А
save/Assign_10Assignlayer_3/kernelsave/RestoreV2_10*
T0*
validate_shape(*
use_locking(*!
_class
loc:@layer_3/kernel*
_output_shapes

:d2
x
save/RestoreV2_11/tensor_namesConst*&
valueBBoutput_layer/bias*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
В
save/Assign_11Assignoutput_layer/biassave/RestoreV2_11*
T0*
validate_shape(*
use_locking(*$
_class
loc:@output_layer/bias*
_output_shapes
:
z
save/RestoreV2_12/tensor_namesConst*(
valueBBoutput_layer/kernel*
dtype0*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
К
save/Assign_12Assignoutput_layer/kernelsave/RestoreV2_12*
T0*
validate_shape(*
use_locking(*&
_class
loc:@output_layer/kernel*
_output_shapes

:2
}
save/RestoreV2_13/tensor_namesConst*+
value"B Btraining/Adam/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Р
save/Assign_13Assigntraining/Adam/Variablesave/RestoreV2_13*
T0*
validate_shape(*
use_locking(*)
_class
loc:@training/Adam/Variable*
_output_shapes

:	2

save/RestoreV2_14/tensor_namesConst*-
value$B"Btraining/Adam/Variable_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Р
save/Assign_14Assigntraining/Adam/Variable_1save/RestoreV2_14*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:2

save/RestoreV2_15/tensor_namesConst*.
value%B#Btraining/Adam/Variable_10*
dtype0*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Ц
save/Assign_15Assigntraining/Adam/Variable_10save/RestoreV2_15*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes

:2d

save/RestoreV2_16/tensor_namesConst*.
value%B#Btraining/Adam/Variable_11*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Т
save/Assign_16Assigntraining/Adam/Variable_11save/RestoreV2_16*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:d

save/RestoreV2_17/tensor_namesConst*.
value%B#Btraining/Adam/Variable_12*
dtype0*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ц
save/Assign_17Assigntraining/Adam/Variable_12save/RestoreV2_17*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes

:d2

save/RestoreV2_18/tensor_namesConst*.
value%B#Btraining/Adam/Variable_13*
dtype0*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Т
save/Assign_18Assigntraining/Adam/Variable_13save/RestoreV2_18*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
:2

save/RestoreV2_19/tensor_namesConst*.
value%B#Btraining/Adam/Variable_14*
dtype0*
_output_shapes
:
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
Ц
save/Assign_19Assigntraining/Adam/Variable_14save/RestoreV2_19*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes

:2

save/RestoreV2_20/tensor_namesConst*.
value%B#Btraining/Adam/Variable_15*
dtype0*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
Т
save/Assign_20Assigntraining/Adam/Variable_15save/RestoreV2_20*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:

save/RestoreV2_21/tensor_namesConst*-
value$B"Btraining/Adam/Variable_2*
dtype0*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
Ф
save/Assign_21Assigntraining/Adam/Variable_2save/RestoreV2_21*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes

:2d

save/RestoreV2_22/tensor_namesConst*-
value$B"Btraining/Adam/Variable_3*
dtype0*
_output_shapes
:
k
"save/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
Р
save/Assign_22Assigntraining/Adam/Variable_3save/RestoreV2_22*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:d

save/RestoreV2_23/tensor_namesConst*-
value$B"Btraining/Adam/Variable_4*
dtype0*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
Ф
save/Assign_23Assigntraining/Adam/Variable_4save/RestoreV2_23*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes

:d2

save/RestoreV2_24/tensor_namesConst*-
value$B"Btraining/Adam/Variable_5*
dtype0*
_output_shapes
:
k
"save/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
Р
save/Assign_24Assigntraining/Adam/Variable_5save/RestoreV2_24*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
:2

save/RestoreV2_25/tensor_namesConst*-
value$B"Btraining/Adam/Variable_6*
dtype0*
_output_shapes
:
k
"save/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
Ф
save/Assign_25Assigntraining/Adam/Variable_6save/RestoreV2_25*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes

:2

save/RestoreV2_26/tensor_namesConst*-
value$B"Btraining/Adam/Variable_7*
dtype0*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
Р
save/Assign_26Assigntraining/Adam/Variable_7save/RestoreV2_26*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:

save/RestoreV2_27/tensor_namesConst*-
value$B"Btraining/Adam/Variable_8*
dtype0*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
Ф
save/Assign_27Assigntraining/Adam/Variable_8save/RestoreV2_27*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes

:	2

save/RestoreV2_28/tensor_namesConst*-
value$B"Btraining/Adam/Variable_9*
dtype0*
_output_shapes
:
k
"save/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
Р
save/Assign_28Assigntraining/Adam/Variable_9save/RestoreV2_28*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:2
ћ
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"з
trainable_variablesПМ
@
layer_1/kernel:0layer_1/kernel/Assignlayer_1/kernel/read:0
:
layer_1/bias:0layer_1/bias/Assignlayer_1/bias/read:0
@
layer_2/kernel:0layer_2/kernel/Assignlayer_2/kernel/read:0
:
layer_2/bias:0layer_2/bias/Assignlayer_2/bias/read:0
@
layer_3/kernel:0layer_3/kernel/Assignlayer_3/kernel/read:0
:
layer_3/bias:0layer_3/bias/Assignlayer_3/bias/read:0
O
output_layer/kernel:0output_layer/kernel/Assignoutput_layer/kernel/read:0
I
output_layer/bias:0output_layer/bias/Assignoutput_layer/bias/read:0
C
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:0
+
	Adam/lr:0Adam/lr/AssignAdam/lr/read:0
7
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:0
7
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:0
4
Adam/decay:0Adam/decay/AssignAdam/decay/read:0
X
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:0
^
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:0
^
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:0
^
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:0
^
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:0
^
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:0
^
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:0
^
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:0
^
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:0
^
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:0
a
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:0
a
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:0
a
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:0
a
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:0
a
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:0
a
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:0"Э
	variablesПМ
@
layer_1/kernel:0layer_1/kernel/Assignlayer_1/kernel/read:0
:
layer_1/bias:0layer_1/bias/Assignlayer_1/bias/read:0
@
layer_2/kernel:0layer_2/kernel/Assignlayer_2/kernel/read:0
:
layer_2/bias:0layer_2/bias/Assignlayer_2/bias/read:0
@
layer_3/kernel:0layer_3/kernel/Assignlayer_3/kernel/read:0
:
layer_3/bias:0layer_3/bias/Assignlayer_3/bias/read:0
O
output_layer/kernel:0output_layer/kernel/Assignoutput_layer/kernel/read:0
I
output_layer/bias:0output_layer/bias/Assignoutput_layer/bias/read:0
C
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:0
+
	Adam/lr:0Adam/lr/AssignAdam/lr/read:0
7
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:0
7
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:0
4
Adam/decay:0Adam/decay/AssignAdam/decay/read:0
X
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:0
^
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:0
^
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:0
^
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:0
^
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:0
^
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:0
^
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:0
^
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:0
^
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:0
^
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:0
a
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:0
a
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:0
a
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:0
a
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:0
a
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:0
a
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:0*
serving_default
/
input&
layer_1_input:0џџџџџџџџџ	9
earnings-
output_layer/BiasAdd:0џџџџџџџџџtensorflow/serving/predict