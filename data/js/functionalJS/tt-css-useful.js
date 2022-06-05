
(function ($, undefined) {
    $(document).ready(function(){

        //vertical align arrow
        $('.right-rail-arrow i.icon-angle-right').each(function(){
            $(this).css("line-height",$(this).closest('.right-rail-arrow').height() +'px')
        })
    })
})(jQuery);
