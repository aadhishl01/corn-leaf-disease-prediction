import { useContext, useEffect, useState } from 'react';
import { assets } from '../assets/assets';
import { ShopContext } from '../context/shopcontext';
import { useLocation } from 'react-router-dom';

const Searchbar = () => {
    const { search, setsearch, setshowsearch, showsearch } = useContext(ShopContext);
    const [visible, setVisible] = useState(false);
    const location=useLocation();
useEffect(() => {
if (location.pathname.includes('/collection')) {
setVisible(true);

}else{
setVisible(false);
}
},[location]);

    return showsearch && visible ? (
        <div className='border-t border-b bg-gray-100 text-center'>
            <div className='inline-flex items-center justify-center border border-gray-400 px-5 py-2 my-5 mx-3 rounded-full sm:w-1/2'>
                <input value={search} onChange={(e) => setsearch(e.target.value)} className='flex-1 outline-none bg-inherit text-sm' type="text" placeholder='search' />
                <img className='w-4' src={assets.search_icon} alt="" />
            </div>
            <img onClick={() => setshowsearch(false)} className='inline w-3 cursor-pointer' src={assets.cross_icon} alt="" />
        </div>
    ) : null
}

export default Searchbar;
