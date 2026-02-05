import { useContext, useEffect, useState } from 'react'
import { ShopContext } from '../context/shopcontext'
import Title from '../components/Title'
import { db, auth } from '../config/firebase'
import { collection, query, where, getDocs } from 'firebase/firestore'
import { onAuthStateChanged } from 'firebase/auth'

const Order = () => {
  const { currency } = useContext(ShopContext);
  const [orderData, setOrderData] = useState([]);

  useEffect(() => {
    // We can use the auth listener here or access 'user' from context.
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (user) {
        loadOrderData(user.uid);
      } else {
        setOrderData([]);
      }
    });
    return () => unsubscribe();
  }, []);

  const loadOrderData = async (uid) => {
    try {
      if (!uid) {
        return null;
      }
      const ref = collection(db, "orders");
      const q = query(ref, where("userId", "==", uid));
      const querySnapshot = await getDocs(q);

      // Flatten the data: each item in an order becomes a row
      let allOrdersItem = [];
      querySnapshot.forEach((doc) => {
        const data = doc.data();
        data.items.forEach((item) => {
          // Add order-level info to each item for display clarity
          const itemWithMeta = {
            ...item,
            status: data.status,
            payment: data.payment,
            paymentMethod: data.paymentMethod,
            date: data.date,
            orderId: doc.id
          };
          allOrdersItem.push(itemWithMeta);
        });
      });
      setOrderData(allOrdersItem.reverse()); // Show newest first
    } catch (error) {
      console.error("Error loading orders:", error);
    }
  }

  return (
    <div className='border-t pt-16'>
      <div className='text-2xl'>
        <Title text1={'MY'} text2={'ORDERS'} />
      </div>

      <div>
        {
          orderData.map((item, index) => (
            <div
              key={index}
              className='py-4 border-t border-b text-gray-700 flex flex-col md:flex-row md:items-center md:justify-between gap-4'
            >
              <div className='flex items-start gap-6 text-sm'>
                <img className='w-16 sm:w-20' src={item.image[0]} alt="" />
                <div>
                  <p className='sm:text-base font-medium'>{item.name}</p>
                  <div className='flex items-center gap-3 mt-2 text-base text-gray-700'>
                    <p className='text-lg'>{currency}{item.price}</p>
                    <p>Quantity: {item.quantity}</p>
                    <p>Size: {item.size}</p>
                  </div>
                  <p className='mt-2'>Date: <span className='text-gray-400'>{new Date(item.date).toDateString()}</span></p>
                  <p className='mt-2'>Payment: <span className='text-gray-400'>{item.paymentMethod}</span></p>
                </div>
              </div>
              <div className='md:w-1/2 flex justify-between'>
                <div className='flex items-center gap-2 justify-center'>
                  <p className='min-w-2 h-2 rounded-full bg-green-500'></p>
                  <p className='text-sm md:text-base'>{item.status}</p>
                </div>
                <div className='flex items-center gap-10'>
                  <button onClick={loadOrderData} className='border border-gray-300 px-4 py-2 text-sm font-medium rounded-sm'>Track Order</button>
                </div>
              </div>
            </div>
          ))
        }
      </div>
    </div>
  )
}

export default Order
