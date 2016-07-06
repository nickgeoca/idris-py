* Create TF library
* Create NN library
* README
* 

1. Is it easier to create new function names with default values that 95% of the time people use? This would reduce clutter.
    * ex tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
        * reduce_mean   : Tensor xs dt -> List Int -> Bool -> Tensor xs dt
        * reduce_mean'  : Tensor xs dt -> List Int -> Tensor xs dt
        * reduce_mean'' : Tensor xs dt -> Tensor [] dt
        * reduce_mean  tensor [1,2] False
        * redue_mean'  tensor [1,2]  
        * redue_mean'' tensor

2. What is the best way to do broadcasting multiplication?
    * New function/operator?
        * mul' x y
        * x *. y
    * Use existing functions, but it will lead to more complex function types. And is easier to make mistakes.
        * mul x y
        * x * y

3. What are the best operators to use?
    * Matlab ex
        * matmul- * 
        * mul- *.
    * TensorFLow ex
        * mul- *
        * matmul- ??

4. How does one make tests?

5. Code standards
** Is camel case better to use? Python community uses lower case, but Idris uses camel case

6. Sort and organize tensorflow functions

7. What is the best way to organize the library files?

8. How can best improve the readability of the last term?
reduce_mean : Tensor shape dt
           -> (remove_dims : List Nat)
           -> (b_keep_rank : Bool)
           -> Tensor (reduce_reshape b_keep_tensor_rank remove_dims shape) dt

9. Fix TensorFlow.Matrix.idr kludge for while_loop.
* The while_loop body/condition functions are parameterized with tensors, but using Idris seems to limit to only being able to parameterize w/ a list of tensors
** Ex: tensorflow python condition(t0,t1,t2,..) ; tensorflow idris condition([t0,t1,t2,..])
* Handling by using a kludge.py python file that does the python *args trick    

10. Constrain matrix operations to appropriate types. E.g. can't do tf.floordiv on complex types