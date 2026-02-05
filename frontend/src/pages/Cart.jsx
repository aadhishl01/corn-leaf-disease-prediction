import { useContext, useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { ShopContext } from '../context/shopcontext.jsx'
import Title from '../components/Title.jsx'
import CartTotal from '../components/CartTotal.jsx'
import { assets } from '../assets/assets.js'

const Cart = () => {
  const { products, currency, cartItems,updateQuantity } = useContext(ShopContext);
  const navigate = useNavigate();
  const [cartdata, setcartdata] = useState([]);

  useEffect(() => {
    const tempdata = [];
    for (const items in cartItems) {
      for (const item in cartItems[items]) {
        if (cartItems[items][item] > 0) {
          tempdata.push({
            _id: items,
            size: item,
            quantity: cartItems[items][item]
          });
        }
      }
    }
    setcartdata(tempdata);
  }, [cartItems]);

  return (
    <div className='border-t pt-14'>
      <div className='text-2xl mb-3'>
        <Title text1={'Your'} text2={'Cart'} />
      </div>

      <div>
        {
          cartdata.map((item, index) => {
            const productData = products.find((product) => product._id === item._id);

            return (
              <div key={index} className='py-4 border-t border-b text-gray-700 grid grid-cols-[4fr_2fr_0.5fr] sm:grid-cols-[4fr_2fr_0.5fr] items-center gap-4'>
                <div className='flex items-start gap-6'>
                  <img className='w-16 sm:w-20' src={productData.image[0]} alt="" />
                  <div>
                    <p className='text-xs sm:text-lg font-medium'>
                      {productData.name}
                    </p>
                    <div className='flex items-center gap-2 mt-1'>
                      <div className='w-4 h-4 rounded-full border' style={{ backgroundColor: productData.color }}></div>
                      <p className='text-xs sm:text-sm text-gray-500'>
                        Size: {item.size} x {item.quantity}
                      </p>
                    </div>
                  </div>
                </div>
                <div className='flex items-center gap-5'>
                  <p>{currency}{productData.price}</p>
                  <p className='px-2 sm:px-3 sm:py-1 border bg-slate-50'>{item.size}</p>
                </div>
                <div className='flex items-center gap-2'>
                  <input onChange={(e)=>e.target.value === '' || e.target.value === '0' ? null :updateQuantity(item._id,item.size,Number(e.target.value))} className='border max-w-10 sm:max-w-20 px-1 sm:px-2 py-1' type="number" min="1" defaultValue={item.quantity} />
                  <img onClick={() => updateQuantity(item._id, item.size, 0)} className='w-4 sm:w-5 cursor-pointer' src={assets.bin_icon} alt='' />
                </div>
              </div>
            )
          })
        }
      </div>
      <div className='flex justify-end my-20'>
        <div className='w-full sm:w-[450px]'>
          <CartTotal />
          <div className='w-full text-end'>
            <button  onClick={()=>navigate('/place-order')}className='bg-black text-white text-sm my-8 px-8 py-3 '>PROCEED TO CHECKOUT</button>
          </div>
        </div>

      </div>
    </div>
  );
}

export default Cart;
