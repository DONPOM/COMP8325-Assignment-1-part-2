
function refreshCategory() {
	//sublist refine response
	if (!$j('.css0').length) {
		$j('.category_sublist_spacer').addClass('css1');
		$j('#category_sublist').prepend('<div class="category_sublist_spacer css0"><div class="refiner">Refine <span class="ref-plus">+</span></div></div>');
		$j('.refiner').click(function() {
			$j('.css1').slideUp();
			if ($j(this).parent().next().is(':hidden')) {
				$j(this).parent().next().slideDown();
			}
			return false;
		});
		//brands
		if ($j('#description .brand_image').length) {
			$j('.css1').prepend('<div class="new-brand-case"></div>')
			$j('#description .brand_image').appendTo('.new-brand-case');
			$j('.new-brand-case img ').addClass('tc_rsp');
		}
	}
	//
	// magic products
	if ($j('.p_t').length) {
		$j('.p_t').remove();
		//console.log('p_t remnants were removed');
	}
	if ($j('.product.item').length) {
		//
		var coolProds=["2","3"];
		for (var i = 0; i < coolProds.length; i++) {
			var pr_t = 0;
			var p_cap = coolProds[i];
			$j('#products').append('<p class="debugg tc_ghost">BEGIN TABLE: '+p_cap+'</p><table cellspacing="10" cellpadding="0" class="debugg p_t p_t_'+p_cap+' feed_t"></table>');
			$j('#products .product.item').each(function() {
				if (pr_t == 0) {
					$j('.feed_t').append('<tr class="feed_r"></tr>');
				}
				$j('.feed_r').append('<td class="prdct product_'+pr_t+' feed_td"><div class="product_thumb_container"><div class="product item_zoom"></div></div><div class="prdct-name"></div><div class="product-tags clearfix"></div></td>');
				$j(this).find('.product_thumb_a').clone().prependTo('.feed_td .product_thumb_container');
				$j(this).find('.item_zoom a').clone().appendTo('.feed_td .item_zoom');
				$j(this).find('.brand_name').clone().prependTo('.feed_td .prdct-name');
				$j(this).find('.product_name_link').clone().appendTo('.feed_td .prdct-name');
				if ($j(this).find('.product-tags').length) {
					$j('.feed_td .product-tags').html($j(this).find('.product-tags').html());
				}
				$j(this).find('.pricing').clone().appendTo('.feed_td');
				$j(this).find('.rating_set').clone().appendTo('.feed_td');
				//special treats
				if ($j(this).find('.product-tag-name-add-to-cart-for-price').length) {
					$j('.feed_td').addClass('special-price');
				}
				//
				$j('.feed_td').removeClass('feed_td');
				pr_t = pr_t+1;
				if (pr_t == p_cap) {
					pr_t = 0;
					$j('.feed_r').removeClass('feed_r');
				}
			});
			//
			var desc = 0;
			if (pr_t) {
				desc = p_cap-pr_t;
			}
			//console.log('descrepency: '+desc+' on cap:'+p_cap);
			if (desc && pr_t) {
				for (var l = 0; l < desc; l++) {
					$j('.feed_r').append('<td class="prdct i-hold-weight"></td>');
				}
			}
			//
			$j('.feed_t').removeClass('feed_t');
			$j('.feed_r').removeClass('feed_r');
		}
		$j('.feed_r').removeClass('feed_r');
		//
		//
		/*$j('img[original]').lazyload({
			effect: 'fadeIn',
			placeholder: "/Public/Images/white.gif"
		});*/
	}
	//
	//$j('.debugg').remove();
}

function resizeCategory() {
	// left vs right
	if ($j('#category_sublist').css('position') == 'absolute') {
		var left = $j('.css1').height() + 20;
		$j('#product_content.with-list').css('min-height',left+'px');
	} else {
		$j('#product_content.with-list').css('min-height','0');
	}
}
//

var res;
window.onresize=function() {
    if (res){clearTimeout(res)};
    res = setTimeout(function(){
		//console.log("resize triggered");
		//category
		resizeCategory();
		//
	},100);
};

$j(document).ready(function(){
	$j('.searchBox').appendTo('.header_spacer');
	$j('.memberBox').appendTo('.header_spacer');
	$j('.shippingBar').appendTo('.header_spacer');
	$j('.header_spacer').append('<div id="mmobi"><a href="#mblmn"></a></div>');
	//
	$j('#meu li.shop_tab').each(function() {
		jQuery('#mblmn ul:first').append('<li class="feed-me mblshp"></li>');
		jQuery('.feed-me').append(jQuery(this).find('a.shop_tab').clone());
		if (jQuery(this).find('div ul:first').length) {
			jQuery('.feed-me').addClass('b-tree');
			jQuery('.feed-me').append(jQuery(this).find('div ul:first').clone());
		}
		jQuery('.feed-me ul, .feed-me ul li').attr('id','');
		jQuery('.feed-me').removeClass('feed-me');
	});
	if (!$j('.ie6').length && !$j('.ie7').length && !$j('.ie8').length) {
		$j('#mblmn').mmenu();
	}
});
