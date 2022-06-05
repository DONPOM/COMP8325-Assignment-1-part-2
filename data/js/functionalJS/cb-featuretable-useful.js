
jQuery(window).load(function(){

//Feature table left bar fix
jQuery('.blk-br').closest('li').addClass('blk-br-add');
jQuery('.red-br').closest('li').addClass('red-br-add');

//Mobile Menus fix
jQuery('#navbar').find('select').find('option:first-child').html('Menu');
jQuery('#top-bar-nav').find('select').find('option:first-child').remove();
var mainmenu = jQuery('#navbar').find('select').html();
jQuery('#top-bar-nav').find('select').prepend(mainmenu);

});
