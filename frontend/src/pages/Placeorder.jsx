import { useContext, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Title from '../components/Title.jsx'
import CartTotal from '../components/carttotal'
import { assets } from '../assets/assets.js'
import { ShopContext } from '../context/shopcontext.jsx'
import { collection, addDoc, Timestamp, doc, getDoc, updateDoc } from 'firebase/firestore'
import { db } from '../config/firebase'
import toast from 'react-hot-toast'

const Placeorder = () => {
  const [method, setMethod] = useState('cod');
  const navigate = useNavigate();
  const { cartItems, setCartItems, getCartAmount, delivery_fee, products, user } = useContext(ShopContext);

  const [formData, setFormData] = useState({
    firstName: '', lastName: '', email: '',
    street: '', city: '', state: '', zipcode: '', country: '', phone: ''
  });

  // Load saved address on mount
  useEffect(() => {
    const loadAddress = async () => {
      if (user) {
        const userDocRef = doc(db, "users", user.uid);
        const userDoc = await getDoc(userDocRef);
        if (userDoc.exists() && userDoc.data().address) {
          setFormData(userDoc.data().address);
        }
      }
    };
    loadAddress();
  }, [user]);

  const onChangeHandler = (event) => {
    const { name, value } = event.target;
    setFormData(data => ({ ...data, [name]: value }));
  }

  const onSubmitHandler = async (event) => {
    event.preventDefault();
    try {
      if (!user) {
        toast.error("Please login to place an order");
        return;
      }

      let orderItems = [];
      for (const items in cartItems) {
        for (const item in cartItems[items]) {
          if (cartItems[items][item] > 0) {
            const itemInfo = structuredClone(products.find(product => product._id === items));
            if (itemInfo) {
              itemInfo.size = item;
              itemInfo.quantity = cartItems[items][item];
              orderItems.push(itemInfo);
            }
          }
        }
      }

      let orderData = {
        userId: user.uid,
        items: orderItems,
        amount: getCartAmount() + delivery_fee,
        address: formData,
        status: "Order Placed",
        paymentMethod: method,
        payment: method === 'cod' ? false : true,
        date: Date.now()
      }

      // Add to Firestore orders
      await addDoc(collection(db, 'orders'), orderData);

      // Update User Profile with new Address
      const userDocRef = doc(db, "users", user.uid);
      await updateDoc(userDocRef, { address: formData });

      setCartItems({});
      toast.success("Order Placed Successfully");
      navigate('/Order');

    } catch (error) {
      console.error(error);
      toast.error(error.message);
    }
  }

  return (
    <form onSubmit={onSubmitHandler} className='flex flex-col sm:flex-row justify-between gap-5 sm:pt-14 min-h-[80vh] border-t'>
      <div className='flex flex-col gap-4 w-full sm:max-w-[480px]'>

        <div className='text-xl sm:text-2xl my-3'>
          <Title text1={'DELIVERY'} text2={'INFORMATION'} />
        </div>

        <div className='flex gap-3'>
          <input required name='firstName' onChange={onChangeHandler} value={formData.firstName} className='border border-gray-300 rounded py-1.5 px-3.5 w-1/2' type="text" placeholder='First Name' />
          <input required name='lastName' onChange={onChangeHandler} value={formData.lastName} className='border border-gray-300 rounded py-1.5 px-3.5 w-full' type="text" placeholder='Last Name' />
        </div>

        <input required name='email' onChange={onChangeHandler} value={formData.email} className='border border-gray-300 rounded py-1.5 px-3.5 w-full' type="email" placeholder='Email Address' />
        <input required name='street' onChange={onChangeHandler} value={formData.street} className='border border-gray-300 rounded py-1.5 px-3.5 w-full' type="text" placeholder='Street' />

        <div className='flex gap-3'>
          <input required name='city' onChange={onChangeHandler} value={formData.city} className='border border-gray-300 rounded py-1.5 px-3.5 w-1/2' type="text" placeholder='City' />
          <input required name='state' onChange={onChangeHandler} value={formData.state} className='border border-gray-300 rounded py-1.5 px-3.5 w-full' type="text" placeholder='State' />
        </div>

        <div className='flex gap-3'>
          <input required name='zipcode' onChange={onChangeHandler} value={formData.zipcode} className='border border-gray-300 rounded py-1.5 px-3.5 w-1/2' type="number" placeholder='Zipcode' />
          <input required name='country' onChange={onChangeHandler} value={formData.country} className='border border-gray-300 rounded py-1.5 px-3.5 w-full' type="text" placeholder='Country' />
        </div>

        <input required name='phone' onChange={onChangeHandler} value={formData.phone} className='border border-gray-300 rounded py-1.5 px-3.5 w-1/2' type="number" placeholder='Phone' />
      </div>

      <div className='mt-8'>
        <div className='mt-8 min-w-80'></div>
        <CartTotal />
      </div>

      <div className='mt-12'>
        <Title text1={'PAYMENT'} text2={'METHOD'} />
        <div className='flex gap-3 flex-col lg:flex-row'>
          <div onClick={() => setMethod('stripe')} className={`flex items-center gap-3 border p-2 px-3 cursor-pointer ${method === 'stripe' ? 'border-green-400 bg-green-50' : ''}`}>
            <p className={`min-w-3.5 h-3.5 border rounded-full ${method === 'stripe' ? 'bg-green-400' : ''}`}></p>
            <img className='h-5 mx-4' src={assets.stripe_logo} alt="" />
          </div>

          <div onClick={() => setMethod('razorpay')} className={`flex items-center gap-3 border p-2 px-3 cursor-pointer ${method === 'razorpay' ? 'border-green-400 bg-green-50' : ''}`}>
            <p className={`min-w-3.5 h-3.5 border rounded-full ${method === 'razorpay' ? 'bg-green-400' : ''}`}></p>
            <img className='h-5 mx-4' src={assets.razorpay_logo} alt="" />
          </div>

          <div onClick={() => setMethod('cod')} className={`flex items-center gap-3 border p-2 px-3 cursor-pointer ${method === 'cod' ? 'border-green-400 bg-green-50' : ''}`}>
            <p className={`min-w-3.5 h-3.5 border rounded-full ${method === 'cod' ? 'bg-green-400' : ''}`}></p>
            <p className='text-gray-500 text-sm font-medium mx-4'>
              CASH ON DELIVERY
            </p>
          </div>
        </div>

        <div className='w-full text-end mt-8'>
          <button type='submit' className='bg-black text-white px-16 py-3 text-sm'>PLACE ORDER</button>
        </div>
      </div>
    </form>
  )
}

export default Placeorder
