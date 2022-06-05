
    (function ($, undefined) {

        var isMobile = {
            Android: function() {
                return navigator.userAgent.match(/Android/i);
            },
            BlackBerry: function() {
                return navigator.userAgent.match(/BlackBerry/i);
            },
            iOS: function() {
                return navigator.userAgent.match(/iPhone|iPad|iPod/i);
            },
            Opera: function() {
                return navigator.userAgent.match(/Opera Mini/i);
            },
            Windows: function() {
                return navigator.userAgent.match(/IEMobile/i);
            },
            any: function() {
                return (isMobile.Android() || isMobile.BlackBerry() || isMobile.iOS() || isMobile.Opera() || isMobile.Windows() || ($('body').width() < 768) );
            }
        };

        var LayoutListener = {

            init : function(){

                if(typeof in_admin == "undefined"){
                    return
                }

                if(!in_admin || in_admin == "false"){
                    return
                }

                if($('#id_snippet').find(":selected").text() == "Custom Promo" ||
                    $('#id_snippet').find(":selected").text() == "Explorer Custom Promo"){
                    this.disableMerlinDelete();
                    this.bindChange();
                    this.bindSubmit();
                    this.chooseLayout();
                    this.hideUnnecessaryFields();
                    this.createHelpButton();
                    // this.enablePMP();

                }
            },

            enablePMP: function(){
                $('.pmp_launcher').css('visibility', 'visible');
            },

            disableMerlinDelete : function(){
                if(typeof in_admin == "undefined"){
                    return
                }

                if(!in_admin || in_admin == "false"){
                    return
                }
                $('.deletelink-box').remove();
                $('.addLink').remove();
                $('body').append("<style type='text/css'>.deletelink-box,.addLink{display:none !important}</style>")
            },

            bindChange : function(){
                var self = this

                var layoutDropDown = $('#var_layout');

                layoutDropDown.change(function (e) {

                    self.chooseLayout()
                });
            },

            bindSubmit : function(){
                form = $('#smartsnippetpointer_form')

                form.submit(function(){
                    LayoutListener.clearHiddenMerlinFields();
                    if(!LayoutListener.validateFields()){
                        return false
                    }
                })
            },

            clearHiddenMerlinFields : function(){

                var fields = $("tr[class^='section_topic']")
                fields.each(function(){
                    if(!$(this).is(":visible")){
                        $(this).find("input").each(function(){
                            $(this).attr("value","")
                        })
                    }
                })
            },

            hideUnnecessaryFields : function(){
                var labels = $('label').filter(function(index) {
                    return $(this).text() == "description" || $(this).text() == "published"
                })

                labels.each(function(elem){
                    $(this).parent().parent().remove()
                })
            },

            validateFields : function(){
                var valid = true
                var fields = $("tr[class^='section_topic']")

                var display_content_channel = $('#var_display_content_channel').find(":selected").text()
                var display_details_texts = $('#var_display_details_texts').find(":selected").text()
                var display_images = $('#var_display_images').find(":selected").text()
                var display_play_icons = $('#var_display_play_icons').find(":selected").text()

                function checkInputChange(elem){
                    $(this).bind('keydown keypress', function() {
                        setTimeout(function() {
                           if(elem.val() != ""){
                                elem.css("border","1px solid #ccc");
                                elem.removeClass('error');
                            }else{
                                elem.css("border","1px solid red");
                                elem.addClass('error');
                            }
                        }, 0);
                    });
                }

                function toggleMinimizeMerlin(elem, open){
                    if(!elem) return false;
                    elem = $(elem);

                    if(open){
                        elem.closest('table').removeClass('closed');
                    }else{
                        elem.closest('table').toggleClass('closed');
                    }

                    if(elem.closest('table').hasClass('closed')){
                        elem.closest('table').find('tr.field').css("display", "none");
                    }else{
                        elem.closest('table').find('tr.field').css("display", "table-row");
                    }
                }

                fields.each(function(){
                    if($(this).is(":visible")){
                        $(this).find("input").each(function(){
                            $(this).css("margin","0");
                            $(this).css("border","1px solid #ccc");
                            $(this).removeClass('error');

                            if($(this).val() === "" ){

                                if(display_content_channel == "Yes" && $(this).attr("id").match("_contentchannel")){
                                    valid = false
                                    $(this).css("border","1px solid red");
                                    $(this).addClass('error');
                                    checkInputChange($(this));
                                }

                                if(display_details_texts == "Yes" && $(this).attr("id").match("_details_text")){
                                    valid = false
                                    $(this).css("border","1px solid red");
                                    $(this).addClass('error');
                                    checkInputChange($(this))
                                }

                                if(display_images == "Yes" && $(this).attr("id").match("_image")){
                                    valid = false;
                                    $(this).css("border","1px solid red");
                                    $(this).addClass('error');
                                    checkInputChange($(this))
                                }

                                if(display_play_icons == "Yes" && $(this).attr("id").match("_image")){
                                    valid = false;
                                    $(this).css("border","1px solid red");
                                    $(this).addClass('error');
                                    checkInputChange($(this))
                                }

                                if($(this).attr("id").match("_short_description") ||
                                    $(this).attr("id").match("_title") ||
                                    $(this).attr("id").match("_url") ||
                                    $(this).attr("id").match("_webobject_type")){
                                    valid = false;
                                    $(this).css("border","1px solid red");
                                    $(this).addClass('error');
                                    checkInputChange($(this))
                                }
                            }
                        })
                    }
                })

                if(!valid){
                    toggleMinimizeMerlin($('input.error').closest('tr[class^="section_"]')
                                         .find('.minimize'), true);
                    alert("Highlighted fields are mandatory. Please correct the errors.");
                }

                return valid;
            },

            chooseLayout : function(){
                var fields = $("tr[class^='section_topic']");
                var selection = $('#var_layout').find(":selected").text();
                var range = parseInt(selection.charAt(0));

                if(isNaN(range)){
                    range = 0;
                }

                for(var i=0; i < fields.length; i++){

                    if (i<range) {
                        fields[i].setAttribute("style","");
                    }else{
                        fields[i].setAttribute("style","display:none");
                    }
                }
            },

            createHelpPoopup : function(){
                newWindow = window.open("http://bento.cdn.pbs.org/hostedbento-prod/filer_public/_bento_media/img/explorer-white-promos.png",'Available layouts','height=468,width=826');
            },

            createHelpButton : function(){
                html = '<a id="layout_help_button" class="plugin-help-tooltip" style="float:none;font-size:17px" href="javascript:void(0)" target="_blank"><img src="http://bento.cdn.pbs.org/hostedbento-prod/filer_public/_bento_media/img/help.png"></a>'
                $('#var_layout').after(html);

                if($('#layout_help_button')){
                    $('#layout_help_button').click(function(){
                        LayoutListener.createHelpPoopup();
                    })
                }
            }
        }

         //fix height 100% on horizontal promos
        function setHeights(){

            if($('body').width() > 600 && $('body').attr('class').indexOf("smartsnippets-smartsnippetpointer") == -1 && !$('html').attr('class').match("lt-ie9 lt-ie8")){
                var promos = $('.horizontal .promo')

                promos.each(function(){

                    $(this).css("height","100%")
                })
                promos.each(function(){

                    $(this).css("height",$(this).parent().height() +"px")

                })
            }
        }

        //fix image ratio height
        function setImageMaxHeight(){
            var hRatio = 9/16
            $('.one-promo a.promo-link img:first-child').each(function(){
                var maxHeight = $(this).width() * hRatio
                $(this).css("max-height",maxHeight+"px")
                //$(this).css("height",maxHeight+"px")
                //check if image is available and set default if not
                if($(this).attr("src") == ""){
                    $(this).attr("src","http://bento.cdn.pbs.org/hostedbento-prod/filer_public/_bento_media/img/no-image-available.jpg")
                }
            })
        }

        $(document).ready(function () {

            if(isMobile.any()){
                $('.promo-container').each(function(){
                    $(this).addClass("mobile")
                })
            }
            LayoutListener.init()
            setImageMaxHeight()
            setHeights()
        });

        $(window).resize(function () {
            setImageMaxHeight()
            setHeights()

        });

        $(window).load(function () {
            LayoutListener.disableMerlinDelete()
            setImageMaxHeight()
            setHeights()
         });


    })(jQuery);
