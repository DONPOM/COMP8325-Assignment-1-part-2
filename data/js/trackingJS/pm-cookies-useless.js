
$j(document).ready(function () {
   $j('.field_category_image_product_id').onFocus = null;
             
   if( $j('.field_category_image_product_id').val()==0) $j('.field_category_image_product_id').val('');
   $j('.field_category_image_product_id').watermark('Product Id or SKU');

});



if (document.cookie.indexOf('GuestStateAdmin=') != -1) {
	$j(document.body).css({
		position: 'relative',
		'background-position': 'center 2em',
		'margin-top': '22px'
	});
	$j('#adminmenu').append('<div id="meua"><p><img class="spinner" src="http://www.petmountain.com/Public/Images/spinner.gif">Loading admin menu...</p></ul>')
	$j('#adminmenu').show();
	$j.ajax({
		url: ajaxFormatUrl('_adminmenu'),
		dataType: 'text',
		cache: false,
		complete: function(jqxhr) {
			if (jqxhr && jqxhr.status == 200) {
								$j('#adminmenu').css({top:'-1000px'}).html(jqxhr.responseText).css({top:'0'});
								$j('#adminmenu').css({position:'fixed'});
				$j(document).trigger('AdminMenu.ready');
			} else {
				$j('#adminmenu').hide();
			}
		}
	});



}
var customerrefhovering=false;


