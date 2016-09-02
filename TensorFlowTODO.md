1. Is it easier to create new function names with default values that 95% of the time people use? This would reduce clutter.
    * ex tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
        * reduce_mean   : Tensor xs dt -> List Int -> Bool -> Tensor xs dt
        * reduce_mean'  : Tensor xs dt -> List Int -> Tensor xs dt
        * reduce_mean'' : Tensor xs dt -> Tensor [] dt
        * reduce_mean  tensor [1,2] False
        * redue_mean'  tensor [1,2]  
        * redue_mean'' tensor
????
 def1: reduce_mean @ tensor
 def2: reduce_mean # tensor [1,2]
 defn: reduce_mean tensor [1,2] False
????
have users define shorter functions in where clause
 redcue_mean1 tensor 

  * Note: using namespace acts as overloading. Will probably do this.

2. What is the best way to do broadcasting multiplication?
    * New function/operator?
        * mul' x y
        * x *. y
    * Use existing functions, but it will lead to more complex function types. And is easier to make mistakes.
        * mul x y
        * x * y
    * functional style
        * map (x+) y   (doesn't make sense if shapes are: [10,1] [1,10])

3. What are the best operators to use?
    * Matlab ex
        * matmul- * 
        * broadcast - ?? 
        * mul- .*
        * dot- dot
    * TensorFLow ex
        * matmul- *> 
        * broadcast - *. 
        * mul- *
        * dot- ??

4. How does one make tests?
 * http://docs.idris-lang.org/en/latest/tutorial/testing.html

5. Code standards
    * Is camel case better to use? Python community uses lower case, but Idris uses camel case

6. Sort and organize tensorflow functions

7. What is the best way to organize the library files?

8. How can best improve the readability of the last term?
reduce_mean : Tensor shape dt
           -> (remove_dims : List Nat)
           -> (b_keep_rank : Bool)
           -> Tensor (reduce_reshape b_keep_tensor_rank remove_dims shape) dt

9. Fix TensorFlow.Matrix.idr kludge for while_loop.
    * The while_loop body/condition functions are parameterized with tensors, but using Idris seems to limit to only being able to parameterize w/ a list of tensors
        * Ex: tensorflow python condition(t0,t1,t2,..) ; tensorflow idris condition([t0,t1,t2,..])
    * Handling by using a kludge.py python file that does the python *args trick    

10. Constrain matrix operations to appropriate types. E.g. can't do tf.floordiv on complex types

11. infixl opeators are not working

12. What is the best way to document functions? It might be better if left TF to the reference documentation if their API changes. Also dependent types document the code to some extent.

13. Python API feed_dict types. Right now works best for placeholders, but would like to expand that to the other types. The python API param feed_dict is refered incorrectly as plaeholders.
feed_dict Type
   Key-Type       Value-Type
 * Tensor       : Python scalar, string, list, or numpy ndarray
 * Placeholder  : Python scalar, string, list, or numpy ndarray
 * SparseTensor : SparseTensorValue.
Can also have nested tuples keys-values, but doesn't seem necessary right now.


----------------------------------------------------------------------------------------------------
There is at least one paper that does this. The Deep Recurrent Attention Model ([DRAM](https://arxiv.org/pdf/1412.7755.pdf)) classifies numbers on an image. One of the outputs is an xy location coordinate of an image. The xy location coordinate selects what part of the image to be read next and _is fed into the input_.

If I am not mistaken, it is not possible to use keras to implement this. 

Is there a way to do this using an existing keras model? And will this be supported in the future?

Here is an idea to implement the user facing side:
* This has a disadvantage in that a feedback loop only connects the last layer to the first. I.e. internal feedbacks loops would not be possible
* Another disadvantage is 

```python
feedback_start_state = tf.zeros([1])
def get_feedback_from_output(output): 
    return tf.slice(output, [0, 0], [-1, 1])  # Get the 0th element from the output. This will be fed into the feedback based input
def combine_input_and_feedback(model, feedback, inp): 
    return tf.concat(1, [feedback, inp])

i = Input(batch_shape=(batch_size, feature_dims))
x = Dense(output_dim=20, activation='linear')(x)
x = Dense(output_dim=2, activation='linear', get_feedback_from_output)(x)
model = ModelFeedback( feedback_start_state=feedback_start_state 
                     , get_feedback_from_output=get_feedback_from_output
                     , combine_input_and_feedback=combine_input_and_feedback
                     , input=i
                     , output=x)
```