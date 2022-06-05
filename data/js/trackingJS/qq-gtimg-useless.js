
    (function($){
        var winWidth = $(window).width();
        var winHeight = $(document).height();
        var shadowWidth = (winWidth - 1002)/2;
        $('#followMe').height( winHeight );
        var dm=document.documentMode;
        var ie=!-[1,];
        var ie7=ie&&dm==7;
        var ie8=ie&&dm==8;
        if( ie7||ie8 ){
            $('#fShaowL').width(shadowWidth+1);
        }else{
            $('#fShaowL').width(shadowWidth);
        }
        $('#fShaowR').width(shadowWidth);
        $('#followBottom').height( winHeight - 1572 );
        var img = new Image();
        img.src= 'http://mat1.gtimg.com/www/images/qq2012/guide_k_01.png';
    })(jQuery);
    function followMeBegin(){
        if( $('#QQ_takeover').css('display') == "block" ){
            $('#followMe').css("top",'30px');
            $('body').append('<div class="f_shadow" id="onTopAdMask" style="width:100%;top:0;left:0;height:30px;"></div>');
        }else{
            $('#followMe').css("top",'0');
        }
        $("#followMe").show();
        $('#follow1').addClass('current');
        $('#follow1Btns').show();
        $(window).scrollTop(0);
        var img = new Image();
        img.src= 'http://mat1.gtimg.com/www/images/qq2012/guide_k_02.png';
    }
    function followMeOver(){
        $('#followMe .current').removeClass('current');
        $("#followMe").hide();
        $('#onTopAdMask').remove();
        $('#follow1Btns,#follow2Btns,#follow3Btns,#follow4Btns,#follow5Btns').hide();
        $('#contentLayout01 .contentModBig .hd').removeClass('hdHover').find('.orderLayout').hide();
        $(window).scrollTop(0);
    }
    function followMeStep2(){
        $('#follow1').removeClass('current');
        $('#follow2').addClass('current');
        $('#follow1Btns').hide();
        $('#follow2Btns').show();
        var img = new Image();
        img.src= 'http://mat1.gtimg.com/www/images/qq2012/guide_k_03.png';
    }
    function followMeStep3(){
        $('#follow2').removeClass('current');
        $('#follow3').addClass('current');
        $('#follow2Btns').hide();
        $('#follow3Btns').show();
        $(window).scrollTop(575);
        var img = new Image();
        img.src= 'http://mat1.gtimg.com/www/images/qq2012/guide_k_04.png';
    }
    function followMeStep4(){
        $('#follow3').removeClass('current');
        $('#follow4').addClass('current');
        $('#follow3Btns').hide();
        $('#follow4Btns').show();
        var img = new Image();
        img.src= 'http://mat1.gtimg.com/www/images/qq2012/guide_k_05.png';
    }
    function followMeStep5(){
        $('#follow4').removeClass('current');
        $('#follow5').addClass('current');
        $('#follow4Btns').hide();
        $('#follow5Btns').show();
        $('#contentLayout01 .contentModBig .hd').addClass('hdHover').find('.orderLayout').show();
        $(window).scrollTop(1062);;
    }
