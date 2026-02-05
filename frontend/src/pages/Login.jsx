import { useState, useContext, useEffect } from 'react'
import { auth, db, googleProvider } from '../config/firebase'
import { createUserWithEmailAndPassword, signInWithEmailAndPassword, signInWithPopup } from 'firebase/auth'
import { setDoc, doc, getDoc } from 'firebase/firestore'
import toast from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'
import { ShopContext } from '../context/shopcontext'

const Login = () => {
  const [currentState, setcurrentState] = useState('Login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const navigate = useNavigate();
  const { user } = useContext(ShopContext);

  useEffect(() => {
    if (user) {
      navigate('/');
    }
  }, [user, navigate]);

  const onSubmitHandler = async (event) => {
    event.preventDefault();
    try {
      if (currentState === 'Sign Up') {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;
        // Create user document in Firestore with initial data
        await setDoc(doc(db, "users", user.uid), {
          email: user.email,
          name: name,
          cart: {}
        });
        toast.success("Account created successfully!");
        navigate('/');
      } else {
        await signInWithEmailAndPassword(auth, email, password);
        toast.success("Logged in successfully!");
        navigate('/');
      }
    } catch (error) {
      console.error(error);
      toast.error(error.message);
    }
  }

  const googleSignInHandler = async () => {
    try {
      const result = await signInWithPopup(auth, googleProvider);
      const user = result.user;

      // Check if user exists
      const userDocRef = doc(db, "users", user.uid);
      const userDoc = await getDoc(userDocRef);

      if (!userDoc.exists()) {
        await setDoc(userDocRef, {
          email: user.email,
          name: user.displayName,
          cart: {}
        });
      }

      toast.success("Logged in with Google!");
      navigate('/');
    } catch (error) {
      console.error(error);
      toast.error(error.message);
    }
  }

  return (
    <div className='flex flex-col items-center w-[90%] sm:max-w-96 m-auto mt-14 text-gray-800 gap-4'>
      <form onSubmit={onSubmitHandler} className='flex flex-col items-center w-full gap-4'>
        <div className='inline flex items-center gap-2 mb-2 mt-10'>
          <p className='prata-regular text-3xl'>{currentState}</p>
          <hr className='border-none h-[1.5px] w-8 bg-gray-800' />
        </div>
        {currentState === 'Login' ? '' : <input type="text" className='w-full px-3 py-2 border border-gray-800' placeholder='Name' required value={name} onChange={(e) => setName(e.target.value)} />}
        <input type="email" className='w-full px-3 py-2 border border-gray-800' placeholder='Email' required value={email} onChange={(e) => setEmail(e.target.value)} />
        <input type="password" className='w-full px-3 py-2 border border-gray-800' placeholder='Password' required value={password} onChange={(e) => setPassword(e.target.value)} />
        <div className='w-full flex justify-between text-sm mt-[-8px]'>
          <p className='cursor-pointer'>Forgot your Password?</p>
          {
            currentState === 'Login'
              ? <p onClick={() => setcurrentState('Sign Up')} className='cursor-pointer'> Create Account</p>
              : <p onClick={() => setcurrentState('Login')} className='cursor-pointer'> Login Here</p>
          }

        </div>
        <button className='bg-black text-white font-light px-8 py-2 mt-4'>{currentState === 'Login' ? 'Sign In' : 'Sign Up'}</button>
      </form>
      <button onClick={googleSignInHandler} className='bg-white border border-gray-300 text-gray-700 font-light px-8 py-2 mt-2 w-full flex items-center justify-center gap-2'>
        <img className='w-5' src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" alt="Google" />
        Continue with Google
      </button>
    </div>
  )
}

export default Login