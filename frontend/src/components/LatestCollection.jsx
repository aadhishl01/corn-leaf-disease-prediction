import { useContext, useState, useEffect } from 'react'
import { ShopContext } from '../context/shopcontext'
import Title from './Title'
import ProductItem from './ProductItem'

const LatestCollection = () => {
    const {products} = useContext(ShopContext);
  const [latestProducts,setlatestProducts] = useState([]);
  useEffect(()=>{
setlatestProducts(products.slice(0,10));
  },[])

  return (
    <div className='my-10'>
        <div className='text-center py-8 text-3xl'>
            <Title text1={'LATEST'} text2={'COLLECTIONS'}/>
            <p className='w-3/4 m-auto text-xs sm:text-sm md:text-base text-gray-600'>
            Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.
            </p>
        </div>
        {/*rendering products*/}
    <div className='grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 gap-y-6'>
      {
        latestProducts.map((items,index)=>(
          <ProductItem key={index} id={items._id} image={items.image} name={items.name} price={items.price}/>
        ))
      }
    </div>
</div>

  )
}

export default LatestCollection