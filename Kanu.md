# Kanu
Kanu is a deep learning framework using [Idris](http://www.idris-lang.org/). The benefit of using Idris is that it is a dependently typed language; meaning it is possible to verify matrix/tensor operations at compile time instead of run time.

The library can create a model and train it using Stochastic Gradient Descent (SGD) with the Cross Entropy function.

Building and running the code has its own section.

Currently the project is on hold. There is a bug with idris-py. The Effect/Monad code can not be TCO- so there are possible stack overflows. 

###Pros
 * Using do notation produces clear code
```Idris
model x = do
  -- Sequential 1
  start x
  dense 10
  y1 <- stop

  -- Sequential 2
  start x
  dense 10
  y2 <- stop

  -- Merge
  y1 * y2
  end
```
 * Fewer run time errors over python method


###Cons
 * Compiling the example takes a while: 1m45s
 * Type errors are more challenging to read in dependently typed language
 * A model's type must include the weights of the model. Need to look more into making the type less onerous (maybe type inference).
 * Can't TCO Effect/Monad code

###Notes:
 * There is a tradeoff between type saftey and simplicity. For example, passing the placeholders along in State is type safer, but more cumbersome (see TensorFlow.Matrix.placeholder)
 * Edwin Brady plans to obsolete Effects. This is currently used in the library.
 * Effects permit changing the type of State (monad state does not allow this). This is simlar to arrows and is flexible.

##Build and run
```shell
# Create Idris executable with Haskell using Stack
git clone <repo>
cd <repo>
stack clean
stack build
# move compiled python lib to idris excutable folder
# mv /home/bob/workspace/idris-py/.stack-work/install/x86_64-linux/lts-6.4/7.10.3/bin/idris-codegen-python /home/bob/workspace/Idris-dev/.stack-work/install/x86_64-linux/lts-6.9/7.10.3/bin/

# Build idris lib
cd <repo>/lib
idris --clean python.ipkg && idris --install python.ipkg

# Compile
cd <repo>/examples
rm kanu.py & idris kanu.idr -p python -p effects --codegen python -o kanu.py

# Run
python kanu.py
```

##Example
```python
$ python kanu.py
"Loss : "
[20.382847]

"Model: "
[array([[  3.55792548e-11,   2.14557971e-07,   1.88433492e-12,
          6.27197383e-12,   5.61418768e-04,   1.08802675e-10,
          6.46610127e-11,   1.40625138e-14,   7.12990295e-05,
          8.42174472e-08],
       [  3.55792548e-11,   2.14557971e-07,   1.88433492e-12,
          6.27197383e-12,   5.61418768e-04,   1.08802675e-10,
          6.46610127e-11,   1.40625138e-14,   7.12990295e-05,
          8.42174472e-08]], dtype=float32)]

"Loss : "
[11.870927]

"Model: "
[array([[  2.14324109e-06,   4.41404300e-05,   2.06270613e-07,
          6.72920066e-07,   1.17864111e-05,   5.65029768e-06,
          6.83615326e-06,   1.53957103e-09,   3.41548026e-03,
          8.86743236e-03],
       [  2.14324109e-06,   4.41404300e-05,   2.06270613e-07,
          6.72920066e-07,   1.17864111e-05,   5.65029768e-06,
          6.83615326e-06,   1.53957103e-09,   3.41548026e-03,
          8.86743236e-03]], dtype=float32)]

"Loss : "
[8.3306599]

"Model: "
[array([[  2.84776138e-06,   2.54918225e-02,   1.53473273e-04,
          3.43506341e-04,   7.45387422e-03,   4.63575361e-06,
          2.66314694e-03,   1.15161060e-06,   9.19207196e-06,
          1.77384824e-01],
       [  2.84776138e-06,   2.54918225e-02,   1.53473273e-04,
          3.43506341e-04,   7.45387422e-03,   4.63575361e-06,
          2.66314694e-03,   1.15161060e-06,   9.19207196e-06,
          1.77384824e-01]], dtype=float32)]

"Loss : "
[6.3519807]

"Model: "
[array([[  3.98982578e-04,   2.62311734e-02,   1.66714694e-02,
          2.76229577e-03,   7.96396956e-02,   6.71481597e-04,
          8.09231494e-03,   1.55181260e-04,   7.40251213e-04,
          1.08200393e-05],
       [  3.98982578e-04,   2.62311734e-02,   1.66714694e-02,
          2.76229577e-03,   7.96396956e-02,   6.71481597e-04,
          8.09231494e-03,   1.55181260e-04,   7.40251213e-04,
          1.08200393e-05]], dtype=float32)]

"Loss : "
[5.2695675]

"Model: "
[array([[ 0.00895214,  0.0113333 ,  0.01573368,  0.02834961,  0.00040454,
         0.01760099,  0.01922174,  0.00281656,  0.00214484,  0.00034793],
       [ 0.00895214,  0.0113333 ,  0.01573368,  0.02834961,  0.00040454,
         0.01760099,  0.01922174,  0.00281656,  0.00214484,  0.00034793]], dtype=float32)]

"Loss : "
[4.7725019]

"Model: "
[array([[ 0.01569801,  0.01211646,  0.01161267,  0.00395896,  0.00572137,
         0.01008734,  0.00909473,  0.01109603,  0.01852503,  0.00198866],
       [ 0.01569801,  0.01211646,  0.01161267,  0.00395896,  0.00572137,
         0.01008734,  0.00909473,  0.01109603,  0.01852503,  0.00198866]], dtype=float32)]

"Loss : "
[4.6296778]

"Model: "
[array([[ 0.00671769,  0.01018541,  0.01028542,  0.01221786,  0.01137974,
         0.01144082,  0.01209658,  0.01055028,  0.00648644,  0.00844552],
       [ 0.00671769,  0.01018541,  0.01028542,  0.01221786,  0.01137974,
         0.01144082,  0.01209658,  0.01055028,  0.00648644,  0.00844552]], dtype=float32)]

"Loss : "
[4.609817]

"Model: "
[array([[ 0.01105584,  0.00992148,  0.01003957,  0.00892624,  0.00907977,
         0.00938197,  0.00901177,  0.00989194,  0.01177379,  0.01086152],
       [ 0.01105584,  0.00992148,  0.01003957,  0.00892624,  0.00907977,
         0.00938197,  0.00901177,  0.00989194,  0.01177379,  0.01086152]], dtype=float32)]
```