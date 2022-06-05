 
     var o={ CartSubtotal: '0.00', CartShipping: '6.99', CartTotal: '6.99', CartTotalItems:'0' }
         jQuery('#cart_request_loading').css('display','none');
         QuickCart_ValuesSet(o);
         jQuery('#cart_request_loading').css('display','none').css('z-index','0');
$j('.qc-total').text('$'+o.CartSubtotal);
$j('#shop_cart_link').addClass('tc_ghost');