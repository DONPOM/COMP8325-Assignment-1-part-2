// share_button Module (widget)

define('module/share_button', ['internal/sitebuilder/common/ModuleClassLoader'], function(ModuleClassLoader) {

    var mod = {}, ext = {};
    (function (module, extend) {

        // Show Cover?
        extend.shouldShowCover = function () {
            return false;
        };

        // Widget
        extend.isWidget = true;



        // Module HTML Template
        extend.template = '<div id="share-container">\
        <script type="text/javascript">\
        var site_locale = typeof(webs) != \'undefined\' && webs.locale || "en-US";\
        \
        site_locale = site_locale == \'zz-ZZ\' ? \'de-DE\' : site_locale;\
        \
        var language = site_locale.split(\'-\')[0];\
        \
        var addthis_config = {\
        ui_language: language\
        };\
        </script>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "showFacebook": true,
            "showTwitter": true,
            "showEmail": true,
            "showGooglePlus": true,
            "showMore": true,
            "styleType": "buttons"
        };

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        module.getMinWidth = function() {
            return 220;
        };

        // Register Toolbars
        bldr.toolbar.register('share_button', {
            "icon": "share_button",
            "items": [{
                "type": "dropdown",
                "label": "",
                "name": "styleType",
                "display": "inline",
                "width": 120,
                "options": [{
                    "label": "Numbers",
                    "value": "buttons"
                }, {
                    "label": "Small Icons",
                    "value": "smallIcons"
                }, {
                    "label": "Large Icons",
                    "value": "largeIcons"
                }
                ]
            }, {
                "type": "menu",
                "label": "Settings",
                "menuLabel": "Advanced Settings",
                "menuClassNames": "w-share-button-menu",
                "menu": [{
                    "display": "inline",
                    "label": "Show Like",
                    "type": "switch-bare",
                    "name": "showFacebook"
                }, {
                    "display": "inline",
                    "label": "Show Tweet",
                    "type": "switch-bare",
                    "name": "showTwitter"
                }, {
                    "display": "inline",
                    "label": "Show Email",
                    "type": "switch-bare",
                    "name": "showEmail"
                }, {
                    "display": "inline",
                    "label": "Show Google Plus",
                    "type": "switch-bare",
                    "name": "showGooglePlus"
                }, {
                    "display": "inline",
                    "label": "Show More",
                    "type": "switch-bare",
                    "name": "showMore"
                }
                ]
            }
            ]
        });

        // View JS
        module.oneLoaded = function() {
            require(["http://s7.addthis.com/js/300/addthis_widget.js#domready=1&pubid=webs"]);
        };

        // Edit JS
        var pageController = bldr.pageController,
        pageDom = pageController.dom;

        // pageDom.$doc.ready(function(){
        // 	var s = pageController.doc.createElement("script");
        // 	s.setAttribute("src", "http://s7.addthis.com/js/300/addthis_widget.js#pubid=xa-4eea3eed22dc65fa&async=1");
        // 	s.setAttribute("id", "addthis-script");
        // 	pageDom.$head.append(s);
        // });

        var require = bldr.pageController.win.require,
        moduleConfig = {
            "smallIcons": {
                "svcs": {
                    "Facebook": {
                        "name": "addthis_button_facebook"
                    },
                    "Twitter": {
                        "name": "addthis_button_twitter"
                    },
                    "Email": {
                        "name": "addthis_button_email"
                    },
                    "GooglePlus": {
                        "name": "addthis_button_google_plusone_share",
                        "attrs": [
                        "g:plusone:size='small'"
                        ]
                    },
                    "More": {
                        "name": "addthis_button_compact"
                    }
                },
                "hasCounter": true,
                "classes": [
                "addthis_toolbox",
                "addthis_default_style"
                ]
            },
            "largeIcons": {
                "svcs": {
                    "Facebook": {
                        "name": "addthis_button_facebook"
                    },
                    "Twitter": {
                        "name": "addthis_button_twitter"
                    },
                    "Email": {
                        "name": "addthis_button_email"
                    },
                    "GooglePlus": {
                        "name": "addthis_button_google_plusone_share"
                    },
                    "More": {
                        "name": "addthis_button_compact"
                    }
                },
                "hasCounter": true,
                "classes": [
                "addthis_toolbox",
                "addthis_default_style",
                "addthis_32x32_style"
                ]
            },
            "buttons": {
                "svcs": {
                    "Facebook": {
                        "name": "addthis_button_facebook_like",
                        "attrs": [
                        "fb:like:layout='button_count'"
                        ]
                    },
                    "Twitter": {
                        "name": "addthis_button_tweet"
                    },
                    "Email": {
                        "name": "addthis_button_email"
                    },
                    "GooglePlus": {
                        "name": "addthis_button_google_plusone",
                        "attrs": [
                        "g:plusone:size='medium'"
                        ]
                    },
                    "More": {
                        "name": "addthis_counter addthis_pill_style"
                    }
                },
                "hasCounter": false,
                "classes": [
                "addthis_toolbox",
                "addthis_default_style"
                ]
            }
        };

        $.extend(module, {
            safeOneLoaded: function() {
                var self = this;
                // this.el.parent().parent();
                this.bind("rendered", function() {
                    try {
                        pageDom.$head.find("#addthis-script").ready(function() {
                            require(["http://s7.addthis.com/js/300/addthis_widget.js#async=1&pubid=webs"], function() {
                                if (!this.parent.addthisLoaded) {
                                    pageController.win.addthis.init();
                                    this.parent.addthisLoaded = true;
                                }
                                self.display();
                            });
                        });
                    } catch (e) {
                        webs.log.error(e.message)
                    }
                });
            },
            getData: function() {
                return this.data;
            },
            display: function() {
                var tbx = this.el.parent().parent().find("#share-container"),
                html = "",
                data = this.data,
                selectedSvc = moduleConfig[data.styleType],
                svcs = selectedSvc.svcs,
                classes = selectedSvc.classes;

                $.each(svcs, function(i, o) {
                    var attrs = o.attrs || [],
                    attrStr = "";
                    if (attrs.length > 0) {
                        $.each(attrs, function(i, o) {
                            attrStr += attrs + " ";
                        });
                    }

                    if (data["show" + i]) {
                        html += '<a class="' + o.name + '" ' + attrStr + '></a>';
                    }
                });

                if (selectedSvc.hasCounter) {
                    html += '<a class="addthis_counter addthis_bubble_style"></a>';
                }

                tbx.removeClass();

                if (classes.length > 0) {
                    $.each(classes, function(i, o) {
                        tbx.addClass(o);
                    });
                }

                tbx.html(html);


                pageController.win.addthis.toolbox(tbx[0]);
                pageController.win.addthis.counter('.addthis_counter'); // This is Clearspring's workaround to display the counter everytime. 
            }
        });



    })(mod, ext);

    return ModuleClassLoader.register('share_button', mod, ext);

});


// text Module (custom)

define('module/text', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.text'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="webs-text {{#if font}}w-font-{{font}}{{/if}}" contenteditable="true" data-placeholder="{{#l}}webs.module.text.clicktoedit{{/l}}" spellcheck="true">{{#if html}}{{{html}}}{{else}}<p>{{#l}}webs.module.text.clicktoedit{{/l}}</p>{{/if}}</div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "html": null
        };

        // SubModules

        // Module Styles
        extend.styles = {};
        extend.defaultStyle = extend.styles[''];

        // Register Toolbars

        // View JS


        // Edit JS
        /**
        * Initialize edit mode
        */
        module.edit = function(event) {
            this.controls = bldr.controls.create('text', {
                el: this.el.find('.webs-text'),
                data: this.data,
                toolbarName: 'text_fonts',
                html: this.data.html,
                linksData: this.data.linksData
            });
            this.controls.show();
            var editFrame = top.$("iframe#website")[0],
            editWindow = editFrame.contentWindow || editFrame.contentDocument.parentWindow;
            this.rangy = editWindow.rangy;
        };

        module.getData = function() {
            if (this.controls) {
                if (this.controls.wizzy) {
                    var controlsData = this.controls.getData();
                    this.data.html = controlsData.html;
                    this.data.font = controlsData.font;
                }
                this.data.linksData = this.controls.getLinksData();
            }
            return this.data;
        };

        module.onDeactivate = function() {
            this.getData();
            if (this.controls) 
                this.controls.destroy();
        };

        module.beforeUndo = function() {
            if (this.controls.wizzy) {
                this.sessionRange = this.controls.wizzy.getRange();
            }
        }

        module.afterUndo = function() {
            // FIXME: we ought to actually add the selection info to the undo data,
            // using rangy's build-in saveSelection and restoreSelection. 
            // But we also don't want to persist it. and persist chokes if it's in the data.
            var range = this.sessionRange;

            if (this.controls.wizzy) {
                this.controls.wizzy.setElement(this.el.find('.webs-text'));
            }

            if (!range || !range.isValid()) {
                range = this.rangy.createRange();
                range.selectNodeContents(this.el.find('.webs-text')[0]);
                range.collapse();
            }

            this.rangy.getSelection().addRange(range);
        };

        module.safeOneLoaded = function() {
            var self = this;
            this.bind("rendered", function() {
                self.el.find("a").bind("click", function(e) {
                    e.preventDefault();
                });
            });
        };


    })(mod, ext);

    return ModuleClassLoader.register('text', mod, ext);
});




// image Module (custom)

define('module/image', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.image'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '{{#wrapInFrame}}\
        <div class="webs-image-wrapper-1{{#if frame}} webs-hasframe{{/if}} webs-align-{{align}}" style="width:{{width}}px;height:{{height}}px;">\
        <div data-placeholder-text="{{#l}}webs.module.image.doubleclicktoedit{{/l}}" class="webs-image-crop {{^url}}bldr-placeholder-image bldr-placeholder-element{{/url}}">\
        <img style="width:{{imageWidth}}%;{{#top}}top:{{.}}px;{{/top}}{{#left}}left:{{.}}px;{{/left}}" src="{{#if url}}{{manipulate_image . fitWidth="1920" fitHeight="1920"}}{{else}}http://static.websimages.com/static/projects/finch/images/placeholder_image.jpg{{/if}}">\
        </div>\
        </div>\
        {{/wrapInFrame}}';

        // Module Default Data
        extend.defaultData = {
            "link": null,
            "imageType": null,
            "width": 300,
            "height": 180,
            "align": "center",
            "imageWidth": 100,
            "caption": "image",
            "inBucket": false,
            "changeCallback": null,
            "frame": "default",
            "frameColor": "ffffff"
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS
        module.fancybox = {
            cssPath: webs.props.staticServer + "/static/global/js/fancybox/jquery.fancybox-1.3.4.css",
            jsPath: webs.props.staticServer + "/static/global/js/fancybox/jquery.fancybox-1.3.4.pack"
        };

        var addCCImage = function(parent, events) {
            if (this.data.imageType == 'flickr') {
                try {
                    webs.page.addCCImage({
                        title: this.data.flickr.title,
                        artist: this.data.flickr.ownerName,
                        url: 'http://www.flickr.com/photos/' + this.data.flickr.ownerId + '/' + this.data.flickr.photoId
                    }, parent, events);
                } catch (ex) {
                    webs.log.error('Unable to create ccImage', this.data, ex);
                }
            }
        };

        var resizeImageForMobile = function() {
            var DEFAULT_OUTER_WIDTH = 320; // For use when there is no frame.
            var self = this;
            var frame = self.el.find('.webs-frame').eq(0),
            img = self.el.find('img'),
            currentWidth = frame.length > 0 ? frame.outerWidth() : img.outerWidth(),
            contentWidth = self.el.hasClass('webs-container') ? self.el.width() : self.el.parents('.webs-container').eq(0).width(),
            widthDelta = currentWidth - contentWidth,
            wrapper = self.el.find('.webs-image-wrapper-1'),
            wrapHeight = wrapper.height(),
            wrapWidth = wrapper.width(),
            newWidth = wrapWidth - widthDelta,
            ratio = newWidth / wrapWidth,
            newHeight = wrapHeight * ratio;

            if (ratio < 1) {
                wrapper.add(img).css({
                    width: newWidth,
                    height: newHeight
                });
            }
            self.el.addClass('mobile-visible');
        };

        if (webs && webs.theme && webs.theme.mobile) {
            module.oneLoaded = function() {

                // We need to shrink the image wrapper and the image itself by the amount of overflow
                // To do this, get the outer width of the frame, subtract 280 from it, then use that number
                // to adjust the width of the wrapper and image. Also, get the percentage change so we can
                // properly adjust the height as well.
                resizeImageForMobile.call(this);

                addCCImage.call(this, '#outer .container', 'touch click');
            }
        } else {
            module.oneLoaded = function() {
                addCCImage.call(this);

                if (this.data.linkInfo && this.data.linkInfo.lightbox && this.data.linkInfo.enabled) {
                    if (!webs.fancybox) {
                        try {
                            $("head").append('<link rel="stylesheet" type="text/css" href="' + this.fancybox.cssPath + '"/>');
                            webs.fancybox = true;
                        } catch (ex) {
                            webs.log.error("Unable to setup fancybox", this.data, ex);
                        }
                    }
                    var self = this;
                    require({
                        paths: {
                            fancybox: this.fancybox.jsPath 
                        }
                    },
                    ["fancybox"],
                    function() {
                        self.el.find("a").attr({
                            href: self.data.url,
                            title: self.data.linkInfo.caption 
                        }).addClass('lightbox-link').fancybox();
                    });
                }
            };
        }


        // Edit JS
        module.safeOneLoaded = function() {
            this.imageLoading = new $.Deferred().resolve().promise();
        };

        module.frameWidth = function() {
            var container = this.el,
            frame = container.find(".webs-frame"),
            inner = container.find(".webs-image-wrapper-1");

            if (frame.length === 0) 
                return 0;
            return frame.outerWidth() - inner.outerWidth();
        };

        module.edit = function() {
            var $container = this.el,
            self = this;
            // attach events, etc.
            this.controls = bldr.controls.create('image', {
                container: $container,
                el: '.webs-image-wrapper-1',
                containerWidth: $container.width(),
                data: this.data,
                beforeChange: function() {
                    self.beforeChange();
                },
                frameWidth: function() {
                    return self.frameWidth();
                }
            });

            this.controls.unbind('setFrame');
            this.controls.bind('setFrame', function(slug) {
                var oldWidth = self.data.width + self.frameWidth();
                self.grow(oldWidth);
                self.data.frame = slug;
                self.el.html(self.html());
                self.fit(oldWidth, true);
                self.controls.readAttributes();
            });

            var imageSet = new $.Deferred();
            imageSet.resolve();
            this.imageLoading = imageSet.promise();

            this.controls.unbind('settingImage');
            this.controls.bind('settingImage', function() {
                imageSet = new $.Deferred();
                self.imageLoading = imageSet.promise();
            });

            this.controls.unbind('setImage');
            this.controls.bind('setImage', function() {
                self.setData(self.controls.getData());
                if (self.data.top > 0) 
                    self.data.top = 0;
                self.init({
                    data: self.data,
                    el: self.el
                });
                setTimeout(function() {
                    self.render();
                }, 1);
                imageSet.resolve();
            });
            this.controls.show();
        };

        module.render = function() {
            var self = this;
            this.imageLoading.done(function() {
                var html = self.html();

                self.el.html(html);
                self.el.find("img").bind("load", function() {
                    setTimeout(function() {
                        self.trigger("rendered", html);
                        if (self.isActivated()) 
                            self.edit();
                    }, 1);
                });
            });
            return this;
        };

        module.setData = function(newData) {
            $.extend(this.data, newData);
            for (var key in this.data) {
                if (!(key in newData))
                    delete this.data[key];
            }
        };

        module.fit = function(width, persist) {
            var $container = this.el,
            wrapper = $container.find('.webs-image-wrapper-1'),
            data = this.data,
            diff;

            width = width - this.frameWidth();
            diff = parseInt(data.width, 10) - width;

            if (diff > 0) {

                var newWidth = width,
                newHeight = parseInt(data.height / data.width * newWidth, 10);

                var $img = wrapper.find("img");
                if ($img[0] && !($img.css("left").match(/%/) && $img.css("top").match(/%/))) {
                    var oldImgPos = $img.position();
                    $img.css({
                        left: (oldImgPos.left / wrapper.width() * 100) + "%",
                        top: (oldImgPos.top / wrapper.height() * 100) + "%"
                    });
                }
                wrapper.width(width).height(newHeight);

                if (persist) {
                    data.width = wrapper.width();
                    data.height = newHeight;
                    wrapper.width(data.width).height(data.height);
                    $.extend(data, $img.position());
                    this.dirty();
                }
            } else if (diff < 0) {
                wrapper.width(data.width).height(data.height);
            }
        };

        module.grow = function(newWidth) {
            var $container = this.el,
            data = this.data,
            wrapper = $container.find('.webs-image-wrapper-1'),
            newHeight = data.height / data.width * newWidth;
            wrapper.width(newWidth).height(newHeight);
            data.width = newWidth;
            data.height = newHeight;
        };

        module.getMinWidth = function() {
            // TODO: Calling this.frameWidth() every time is inefficient, but may be necessary.
            var base = 50;
            return base + this.frameWidth();
        };

        module.getData = function() {
            if (this.controls) 
                this.setData(this.controls.getData());
            return this.data;
        };

        module.onDeactivate = function() {
            var self = this;
            this.imageLoading.done(function() {
                if (self.controls) {
                    self.controls.readAttributes();
                    self.setData(self.controls.getData());
                    self.controls.hide();
                    self.controls.unbind('setFrame');
                    delete self.controls;
                }
            });
        };


    })(mod, ext);

    return ModuleClassLoader.register('image', mod, ext);
});




// carousel Module (custom)

define('module/carousel', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.carousel'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return true;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '\
        <div class="w-carousel \
        {{#case "1" slides.length}}single_slide{{/case}}\
        {{#if hideDescription}}hide_description{{/if}}\
        {{#if hideButton}}hide_button{{/if}}\
        {{#if hideText}}hide_text{{/if}}\
        {{#if hideTitle}}hide_title{{/if}}" style="height: {{height}}px;">\
        <div class="w-carousel-list-wrapper">\
        <ul class="w-carousel-list">\
        {{#each slides}}\
        <li {{#case _i_ 0}}class="active"{{/case}} data-index="{{_i_}}">\
        <div class="w-carousel-slide">\
        {{#image}}\
        <div data-placeholder-text="{{#l}}webs.module.carousel.doubleclicktoedit{{/l}}" class="w-carousel-image {{#unless url}}bldr-placeholder-element bldr-carousel-placeholder{{/unless}}">\
        <div class="webs-image-crop {{#unless url}}bldr-placeholder-element bldr-placeholder-carousel{{/unless}}">\
        <img{{#if alt}} alt="{{alt}}"{{/if}} src="{{#if url}}{{#if ../../../mobile_view}}{{manipulate_image . fitWidth="960" fitHeight="960"}}{{else}}{{manipulate_image . fitWidth="1920" fitHeight="1920"}}{{/if}}{{/if}}" style="left: {{left}}px; top: {{top}}px; width: {{width}}%;">\
        </div>\
        </div>\
        {{/image}}\
        <div class="w-carousel-description">\
        <h6 class="w-carousel-title webs-title"><div class="w-text {{#if font}}w-font-{{font}}{{/if}}">{{#if title}}{{{title}}}{{else}}{{#l}}webs.module.carousel.defaultTitle{{/l}}{{/if}}</div></h6>\
        <div class="w-carousel-text w-text">{{#if text}}{{{text}}}{{else}}{{#l}}webs.module.carousel.defaultText{{/l}}{{/if}}</div>\
        </div>\
        </div>\
        </li>\
        {{/each}}\
        </ul>\
        </div>\
        <a href="#" class="w-carousel-prev" title="{{#l}}webs.module.carousel.previous{{/l}}"><span>&lt;</span></a>\
        <ul class="w-carousel-indices">\
        {{#each slides}}\
        <li {{#case _i_ 0}}class="active"{{/case}}><a href="#" class="w-carousel-index"><span>{{_ip1_}}</span></a></li>\
        {{/each}}\
        </ul>\
        <a href="#" class="w-carousel-next" title="{{#l}}webs.module.carousel.next{{/l}}"><span>&gt;</span></a>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "height": 200,
            "hideDescription": true,
            "hideTitle": true,
            "hideText": true,
            "hideButton": true,
            "transition": "carouselHorizontal",
            "autoplay": true,
            "speed": 7000,
            "slides": [{
                "linksData": {},
                "image": {
                    "url": null,
                    "fullWidth": 1969,
                    "fullHeight": 1084,
                    "width": 100,
                    "top": 0,
                    "left": 0
                },
                "button": {
                    "link": null,
                    "bgcolor": null,
                    "align": "left",
                    "linkInfo": {
                        "linkType": "website",
                        "linkUrl": null,
                        "newWindow": false
                    },
                    "icon": {
                        "enable": false,
                        "set": null,
                        "slug": null,
                        "position": "left",
                        "style": null
                    }
                }
            }
            ]
        };

        // SubModules
        extend.submodules = {
            "button": {
                "moduleType": "button"
            }
        };

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "plain.view.less",
                    "js": "plain.view.js"
                },
                "default": true,
                "slug": "default"
            }
        };
        extend.styles['default']['global']['js'] = (function(module, extend) {


            /* plain.view.js */

            return module;
        });
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars
        bldr.toolbar.register('carousel_image', {
            "icon": "image",
            "items": [{
                "type": "button",
                "style": "blue",
                "label": "Change Image",
                "name": "changeImage",
                "dialog": {
                    "url": "/s/sitebuilder/dialogs/image",
                    "localizeHeading": "webs.bldr.toolbar.text.ChangeImage",
                    "width": 650,
                    "height": 361
                }
            }
            ]
        });

        // View JS
        if (webs && webs.theme && webs.theme.mobile) {
            var BURNS_DURATION = 6000,
            FADE_DURATION = 2000,
            SLIDE_DURATION = BURNS_DURATION - FADE_DURATION,
            $html,
            useCssAnim;

            $.extend(module, {
                oneLoaded: function() {
                    var self = this,
                    data = this.data;

                    $html = $('html');
                    useCssAnim = $html.hasClass('cssanimations');

                    jQuery.fx.interval = 33;

                    this.activeIndex = 0;

                    this.$carousel = this.el.find('.w-carousel');
                    this.$list = this.$carousel.find('ul.w-carousel-list');

                    this.$indices = this.$carousel.find('.w-carousel-indices').children();

                    this.$list.children().each(function() {
                        var s = $(this),
                        img = s.find('img'),
                        src = img.attr('src');
                        if (!src) {
                            s.remove();
                        }
                    });

                    this.slides = this.$list.children();

                    this.count = this.slides.length;
                    this.width = this.$carousel.width();

                    if (this.count === 0) {
                        this.$carousel.addClass('empty');
                        return;
                    } else {
                        this.$carousel.addClass('populated');
                    }

                    this.slides.each(function() {
                        var slide = $(this),
                        image = slide.find('img'),
                        mobileImage = image.clone(true, true);

                        mobileImage.removeAttr('style');
                    });

                    this.pushCCInfo();

                    if (this.count > 1) 
                        this.play();
                },
                play: function() {
                    var self = this;
                    this.activeSlide = this.$list.children().eq(this.activeIndex);
                    this.animate(this.activeSlide);
                    setInterval(function() {
                        self.nextSlide();
                    }, SLIDE_DURATION);
                },
                animate: function(slide) {
                    var rand1 = Math.round(Math.random() * 100),
                    rand2 = Math.round(Math.random() * 100),
                    origin = rand1 + '% ' + rand2 + '%',
                    randTop = Math.round(Math.random() * (Math.max(0, slide.find('img').height() - 120)));

                    slide.addClass('active');
                    if (useCssAnim) {
                        slide.find('img').css({
                            '-moz-transform-origin': origin,
                            '-ms-transform-origin': origin,
                            '-webkit-transform-origin': origin,
                            'transform-origin': origin,
                            'top': '-' + randTop + 'px'
                        }).addClass('burns');
                    } else {
                        require([webs.props.staticServer + '/active-static/lib/jquery.kenburns-0.1.js'], function jsAnimate() {
                            slide.find('img').kenBurns({
                                width: this.width,
                                height: 120,
                                duration: BURNS_DURATION
                            });
                        });
                    }
                },
                getNext: function() {
                    var nextIndex = this.activeIndex + 1;
                    var nextSlide = this.activeSlide.next();
                    if (nextSlide.length === 0) {
                        nextIndex = 0;
                        nextSlide = this.$list.children().eq(nextIndex);
                    }
                    return {
                        index: nextIndex,
                        slide: nextSlide
                    };
                },
                fadeOut: function(slide) {
                    var z = slide.css('z-index');

                    slide.css('z-index', '1000').fadeOut(FADE_DURATION, function() {
                        slide.removeClass('active').removeAttr('style').css('z-index', z);
                    });
                },
                nextSlide: function() {
                    var next = this.getNext();

                    this.fadeOut(this.activeSlide);
                    this.animate(next.slide);

                    this.activeIndex = next.index;
                    this.activeSlide = next.slide;
                },

                pushCCInfo: function() {
                    var slides = this.data.slides;
                    for (var i = 0, len = slides.length; i < slides.length; i++) {
                        var image = slides[i].image;
                        if (image.imageType == 'flickr') {
                            try {
                                webs.page.addCCImage({
                                    title: image.flickr.title,
                                    artist: image.flickr.ownerName,
                                    url: 'http://www.flickr.com/photos/' + image.flickr.ownerId + '/' + image.flickr.photoId
                                }, '#outer .container', 'click touch');
                            } catch (ex) {
                                webs.log.error('Unable to create ccImage', image, ex);
                            }
                        }
                    }
                }
            });
        } else {
            require(["internal/sitebuilder/modules/common/transitions"]);
            $.extend(module, {

                events: {
                    'click .w-carousel-prev': 'previousSlide',
                    'click .w-carousel-next': 'nextSlide',
                    'click .w-carousel-index': 'indexClick'
                },

                oneLoaded: function() {
                    var self = this, data = this.data;

                    this.activeSlide = 0; // current slide's index

                    this.$carousel = this.el.find('.w-carousel');
                    this.$list = this.$carousel.find('ul.w-carousel-list');
                    this.$indices = this.$carousel.find('.w-carousel-indices').children();
                    this.slides = this.$list.children();
                    this.count = this.slides.length;

                    this.pushCCInfo();

                    window.temp = $('#webs-header').trigger('slidesLoaded');

                    if (this.data.autoplay && this.count > 1) 
                        this.play();
                },

                play: function() {
                    var self = this;
                    setInterval(function() {
                        self.nextSlide();
                    }, this.data.speed);
                },

                indexClick: function(e) {
                    this.selectSlide(parseInt($(e.target).text(), 10) - 1);
                    return false;
                },

                selectSlide: function(i) {
                    if (i===-1) 
                        i += this.count;
                    else if (this.count === i) 
                        i = 0;

                    if (this.transitioning) 
                        return false;

                    var
                    self = this,
                    oldEl = this.slides.eq(this.activeSlide).addClass('outgoing'),
                    newEl = this.slides.eq(i).addClass('incoming'),
                    oldi = this.activeSlide,
                    transition = webs.transitions[this.data.transition] || webs.transitions.none,
                    invertTransition = (i < oldi) && !(i === 0 && oldi === this.count - 1) ||
                    (i === this.count - 1 && oldi === 0);

                    this.$carousel.addClass("w-carousel-transitioning");

                    this.activeSlide = i;
                    this.transitioning = true;

                    transition(oldEl, newEl, this.data.duration, invertTransition, function() {
                        self.transitioning = false;
                        oldEl.removeClass("outgoing active");
                        newEl.removeClass("incoming").addClass("active");
                        self.$carousel.removeClass("w-carousel-transitioning");

                        self.$indices.eq(oldi).removeClass("active");
                        self.$indices.eq(i).addClass("active");
                    });

                    self.el.trigger('slideSelected', self);
                },
                previousSlide: function() {
                    this.selectSlide(this.activeSlide - 1);
                    return false;
                },
                nextSlide: function() {
                    this.selectSlide(this.activeSlide + 1);
                    return false;
                },

                describeSubmodules: function() {
                    var self = this,
                    buttons = self.el.find(".w-carousel-button");
                    submoduleDescriptions = [];
                    var count = 0;
                    $.each(buttons, function(buttonIndex) {
                        if (self.data.slides.length <= buttonIndex) 
                            return;
                        submoduleDescriptions.push({
                            name: "button" + buttonIndex,
                            el: buttons.eq(buttonIndex).find(".webs-submodule"),
                            slug: "button",
                            data: self.data.slides[buttonIndex].button
                        });
                    });
                    return submoduleDescriptions;
                },

                pushCCInfo: function() {
                    var slides = this.data.slides;
                    for (var i = 0, len = slides.length; i < slides.length; i++) {
                        var image = slides[i].image;
                        if (image.imageType == 'flickr') {
                            try {
                                webs.page.addCCImage({
                                    title: image.flickr.title,
                                    artist: image.flickr.ownerName,
                                    url: 'http://www.flickr.com/photos/' + image.flickr.ownerId + '/' + image.flickr.photoId
                                });
                            } catch (ex) {
                                webs.log.error('Unable to create ccImage', image, ex);
                            }
                        }
                    }
                }
            });
        }

        // Edit JS
        /**
         * This module uses an element pattern where the module creates a controller (module.SlideController) for each slide
         * The $ here is bldr.pageController.$
         */
        var SlideController = (function($) {

            if (typeof Handlebars !== 'undefined') {
                // TODO: This template is duplicated from view.hbs. Changing one means you must change the other. This is bad; let's figure out how to store it once and reference it from both places.
                // Should we allow modules to register their own helpers?
                var slideTemplate = Handlebars.compile('\
                <div class="w-carousel-slide">\
                {{#image}}\
                <div class="w-carousel-image {{#unless url}}bldr-carousel-placeholder{{/unless}}">\
                <div data-placeholder-text="{{#l}}webs.module.carousel.doubleclicktoedit{{/l}}" class="webs-image-crop {{#unless url}}bldr-placeholder-element bldr-placeholder-carousel{{/unless}}">\
                {{#if url}}<img src="{{manipulate_image . fitWidth="1920" fitHeight="1920"}}" style="left: {{left}}px; top: {{top}}px; width: {{width}}%;">{{/if}}\
                </div>\
                </div>\
                {{/image}}\
                <div class="w-carousel-description">\
                <h6 class="w-carousel-title webs-title"><div class="w-text {{#if font}}w-font-{{font}}{{/if}}">{{#if title}}{{{title}}}{{else}}{{#l}}webs.module.carousel.defaultTitle{{/l}}{{/if}}</div></h6>\
                <div class="w-carousel-text">{{#if text}}{{{text}}}{{else}}{{#l}}webs.module.carousel.defaultText{{/l}}{{/if}}</div>\
                </div>\
                </div>');
            }

            /**
            	 * When SlideController.init(opts) is called, the return will be a new instance of Slide Class
            	 * Every key-val in the opts object will become the new SlideClass's instance variable
            	 *
            	 * Controls that alter each specific slide (zoom, pan, title, text, image, etc.) should go here
            	 */
            var SlideClass = function(opts) {
                for (var key in opts) {
                    if (opts.hasOwnProperty(key)) {
                        this[key] = opts[key];
                    }
                }
                this.el = $(this.el[0]);
                if (this.isNew) 
                    this.render();

                this.init(opts);
            };
            SlideClass.prototype = {
                init: function(opts) {
                    var self = this;

                    this.$imageContainer = this.el.find('.w-carousel-image');
                    this.$image = this.$imageContainer.find('img');
                    this.$title = this.el.find('.w-carousel-title');
                    this.$text = this.el.find('.w-carousel-text');

                    this.imageToolbar = bldr.toolbar.create("carousel_image");
                    this.imageToolbar.setFeedback(this.el);
                    this.imageToolbar.bind("changeImage", function(param) {
                        self.changeImage(param);
                    });
                    this.changeImageButton = this.imageToolbar.find(".w-tbui-changeImage");
                    this.el.delegate('.w-carousel-image', "dblclick", function() {
                        self.changeImageButton.click();
                    });
                    this.el.delegate(".w-carousel-image", "click", function() {
                        self.imageToolbar.show();
                    });

                    this.attachControls();

                    this.tmpLeft = this.data.image.left;
                    this.tmpWidth = this.data.image.width;
                    this.data.linksData = this.data.linksData || {};
                },
                attachControls : function() {
                    var self = this,
                    fullWidth = this.el.width();

                    this.titleControls = bldr.controls.create('title', {
                        el: this.$title,
                        toolbarName: 'title_condensed',
                        data: this.data
                    });
                    this.textControls = bldr.controls.create('text', {
                        el: this.$text,
                        toolbarName: 'text_condensed',
                        html: this.data.text,
                        linksData: this.data.linksData
                    });


                    this.$imageContainer.click(function() {
                        var isPlaceholder = self.$imageContainer.children(".bldr-placeholder-element")[0];
                        self.$zoomer[isPlaceholder ? "addClass" : "removeClass"]("inactive");
                    });
                    this.$title.click(function() {
                        self.textControls.hide();
                        self.titleControls.show();
                        self.$zoomer.addClass("inactive");
                    });
                    this.$text.click(function() {
                        self.titleControls.hide();
                        self.textControls.show();
                        self.$zoomer.addClass("inactive");
                    });
                    this.el.click(function(e) {
                        var $target = $(e.target);
                        if (!$target.hasClass("w-carousel-text") && !$target.parents(".w-carousel-text").length) {
                            self.textControls.hide();
                            if ($.trim(self.$text.text()) === "") {
                                self.$text.html(translate("webs.module.carousel.defaultText"));
                            }
                        } else if (!$target.hasClass("w-carousel-title") && !$target.parents(".w-carousel-title").length) {
                            self.titleControls.hide();
                            if ($.trim(self.$title.find(".w-text").text()) === "") {
                                self.$title.find(".w-text").html(translate("webs.module.carousel.defaultTitle"));
                            }
                        }
                    });
                    /*
                    			this.$button = this.el.find('.w-carousel-button');
                    			this.$button.click(function(){
                    				self.titleControls.hide(); self.textControls.hide();
                    				self.$zoomer.addClass("inactive");

                    				var btnController = self.$button.find(".webs-submodule").data("controller");
                    				if(btnController) btnController.attachControls();
                    				return false;
                    			});*/

                    this.$image.zoomer({
                        min: fullWidth,
                        max: fullWidth * 2,
                        reference: this.$imageContainer,
                        dragEnd: function() {
                            self.tmpLeft = self.data.image.left;
                            self.tmpWidth = self.data.image.width;
                        },
                        onChange: function(val) {
                            self.setZoom(parseInt(val, 10), fullWidth);
                        }
                    });
                    this.$zoomer = this.el.find(".bldr-image-zoomer");

                    // prevent quick clicks on +/- on zoomer from opening change image dialog
                    this.el.delegate(".bldr-image-zoomer", "dblclick", function() {
                        return false;
                    });

                    this.$image.panner({
                        reference: this.$imageContainer,
                        onEnd: function(e) {
                            self.setPosition();
                        }
                    });

                },
                setPosition : function() {
                    var p = this.$image.position();
                    this.data.image.top = p.top;
                    this.data.image.left = p.left;
                    this.tmpLeft = p.left;
                },
                setZoom : function(width, fullWidth) {
                    var _image = this.data.image,
                    _width = 100 * width / fullWidth,
                    _left = (fullWidth + ( - this.tmpLeft * 2) - width) / 2;

                    if (width + _left < fullWidth) {
                        _left = fullWidth - width;
                    } else if (_left > 0) {
                        _left = 0;
                    }

                    this.$image.css({
                        width: _width + '%',
                        left: _left + 'px'
                    });
                    _image.width = _width;
                    _image.left = _left;
                },

                changeImage : function(param) {
                    var _img = this.data.image;
                    _img.left = 0;
                    _img.top = 0;
                    _img.width = 100;
                    _img.url = param.url;

                    _img.imageType = param.imageType;
                    if (param[_img.imageType]) {
                        // Copy type-specific info (webs fileId, flickr photoId, etc.)
                        _img[_img.imageType] = param[_img.imageType];
                    }

                    this.saveData().render().init();
                    this.parent.bindSubmodules();
                    this.parent.attachControls();
                    this.$image.click();
                },

                activate : function() {
                    this.mbAnchor.addClass('active');
                    this.el.addClass('active').removeClass('incoming').css('z-index', '');
                    this.parent.attachCropper(this.$image);
                    this.$image.css({
                        "left": this.data.left,
                        "top": this.data.top
                    });
                    this.imageToolbar.show();
                    return this;
                },
                deactivate : function() {
                    this.mbAnchor.removeClass('active');
                    this.el.removeClass('active outgoing');
                    this.titleControls.wizzy.hide();
                    if ($.trim(this.$title.find(".w-text").text()) === "") {
                        this.$title.find(".w-text").html(translate("webs.module.carousel.defaultTitle"));
                    }
                    if ($.trim(this.$text.text()) === "") {
                        this.$text.html(translate("webs.module.carousel.defaultText"));
                    }
                    return this;
                },
                render : function() {
                    this.el.html(slideTemplate(this.data));
                    this.init();
                    this.parent.attachCropper(this.$image);
                    return this;
                },

                destroy : function() {
                    this.mbAnchor.unbind();
                    this.mbButton.remove();
                    this.$image.panner('destroy');
                    this.$image.zoomer('destroy');

                    this.textControls.destroy();
                    this.titleControls.destroy();
                    this.el.remove();
                },

                // saves text data, happens on the module's blur event
                saveData : function() {
                    /*
                    			 var btnController = this.$button.find(".webs-submodule").data("controller");
                    			if(btnController) this.data.button = btnController.getData();
                    			*/
                    if (this.titleControls) {
                        this.data.title = this.titleControls.getData().html;
                    }
                    if (this.textControls) {
                        this.data.text = this.textControls.getData().html;
                    }

                    return this;
                },

                getData : function() {
}
            };
            return {
                init: function(opts) {
                    return new SlideClass(opts);
                }
            };

        })(typeof bldr !== 'undefined' ? bldr.pageController.$ : $);








        $.extend(module, {

            safeOneLoaded: function() {
                $header = this.el.parents('#webs-header');
                var firstRendered = false;
                this.bind('rendered', function() {
                    if (firstRendered) 
                        return;
                    $header.trigger('slidesLoaded');
                    firstRendered = true;
                });
            },
            events: {
                'click .w-carousel-prev': 'previousSlide',
                'click .w-carousel-next': 'nextSlide',
                'click .w-carousel-index': 'indexClick'
            },

            onActivate: function() {
                var self = this,
                data = this.data;

                this.slideControllers = []; // controllers for each slide, in order
                this.activeSlide = 0; // current slide's index

                this.$carousel = this.el.find('.w-carousel');
                this.$listWrap = this.$carousel.find('.w-carousel-list-wrapper');
                this.$list = this.$carousel.find('ul.w-carousel-list');
                this.$indices = this.el.find(".w-carousel-indices");

                this.createModulebar();
                this.$modulebar = this.moduleBar.container.addClass('bldr-modulebar-bottom');
                this.$buttonContainer = this.moduleBar.buttonContainer;
                this.buildSlides(this.data.slides);
                this.selectSlide(0);

                // SITEBUILDER-1393 - remove button option
                data.hideButton = true;

                this.attachCropper();
            },

            attachCropper: function(target) {
                // Controls
                var self = this;
                this.$listWrap.cropper("destroy");
                this.$listWrap.cropper({
                    target: target || this.$carousel.find("img"),
                    sides: ['bottom'],
                    onMove: function(side, width, height) {
                        if (height > 0) {
                            self.setCrop(height);
                            bldr.pageController.showCover();
                        }
                    }
                });
            },


            afterUndo: function() {
                if (this._isActivated) {
                    // coming from undo but the module is already active
                    this.moduleBar.unbind();
                    this.moduleBar.buttonContainer.empty();
                    this.moduleBar.buttonContainerRight.children(':not(.bldr-modulebar-done-container)').remove();
                    this.destroy();
                    this.activate();
                }
            },

            attachCompositeControls: function() {
                $.each(this.submoduleInstances, function(index, submodule) {
                    submodule.attachControls();
                });
            },

            getData: function() {
                if (this.slideControllers) {
                    for (var i = 0, len = this.slideControllers.length; i < len; i++) {
                        this.slideControllers[i].saveData();
                    }
                }
                return this.data;
            },
            destroy: function() {
                for (var i = 0, len = this.slideControllers.length; i < len; i++) {
                    this.slideControllers[i].saveData().destroy();
                }
                this.$carousel.cropper('destroy');
                this.render();
            },


            /**
            	 * Creates the module bar and bind modulebar events directly to the module's methods
            	 */
            createModulebar: function() {
                var self = this, data = this.data;

                this.moduleBar = bldr.pageController.getModuleBar();
                this.moduleBar.build({
                    module: self,
                    width: 180,
                    title: 'Carousel Settings',
                    menu: [
                    {
                        type: 'switch-bare',
                        label: 'Autoplay',
                        name: 'autoplay',
                        display: 'inline',
                        classNames: data.autoplay ? 'active' : null,
                        labelWidth: 70
                    },
                    {
                        type: 'dropdown',
                        label: 'Transition',
                        name: 'transition',
                        width: 100,
                        display: 'inline',
                        labelWidth: 70,
                        selectedValue: data.transition,
                        options: [
                        {
                            label: 'None',
                            value: 'none' 
                        },
                        {
                            label: 'Fade',
                            value: 'fade' 
                        },
                        {
                            label: 'Puff',
                            value: 'puff' 
                        },
                        {
                            label: 'H Slide',
                            value: 'slideHorizontal' 
                        },
                        {
                            label: 'V Slide',
                            value: 'slideVertical' 
                        },
                        {
                            label: 'H Carousel',
                            value: 'carouselHorizontal' 
                        },
                        {
                            label: 'V Carousel',
                            value: 'carouselVertical' 
                        }
                        ]
                    },
                    {
                        type: 'dropdown',
                        label: 'Speed',
                        name: 'speed',
                        width: 100,
                        display: 'inline',
                        labelWidth: 70,
                        selectedValue: data.speed,
                        options: [
                        {
                            label: '3 Sec',
                            value: 3000 
                        },
                        {
                            label: '4 Sec',
                            value: 4000 
                        },
                        {
                            label: '5 Sec',
                            value: 5000 
                        },
                        {
                            label: '6 Sec',
                            value: 6000 
                        },
                        {
                            label: '7 Sec',
                            value: 7000 
                        }
                        ]
                    }
                    ]
                });

                this.$addSlideBtn = this.moduleBar.addButton({
                    label: '+',
                    name: 'addSlide' 
                });
                this.moduleBar.addButton({
                    type: 'divider' 
                });
                this.moduleBar.addButton({
                    icon: 'trash',
                    style: 'red',
                    label: 'Delete Slide',
                    name: 'deleteSlide' 
                });


                this.moduleBar.buttonContainerRight.append(
                this.moduleBar.addButton({
                    type: 'divider'
                }),
                this.moduleBar.addButton({
                    type: 'switch',
                    label: 'Title',
                    name: 'toggleTitle',
                    classNames: !data.hideTitle ? 'active' : null 
                }),
                this.moduleBar.addButton({
                    type: 'switch',
                    label: 'Text',
                    name: 'toggleText',
                    classNames: !data.hideText ? 'active' : null 
                })
                );

                this.moduleBar.bind({
                    autoplay: function(on) {
                        self.setAutoplay(on);
                    },
                    transition: function(t) {
                        self.setTransition(t);
                        self.selectSlide(self.activeSlide + 1);
                    },
                    speed: function(s) {
                        self.setDuration(s);
                    },

                    addSlide: function() {
                        self.addSlide();
                    },
                    deleteSlide: function() {
                        self.deleteSlide();
                    },
                    selectSlide: function(getSelf) {
                        if (typeof getSelf === 'function') 
                            self.selectSlide(getSelf().data('index'));
                    },

                    toggleButton: function(on) {
                        self.toggleButton(on);
                    },
                    toggleText: function(on) {
                        self.toggleText(on);
                    },
                    toggleTitle: function(on) {
                        self.toggleTitle(on);
                    }
                });
                this.moduleBar.setFeedback(this.el);

                this.createModulebarScroller();
            },


            /**
            	 * Scrollbar for the slides in the modulebar when they overflow
            	 */
            createModulebarScroller: function() {
                var self = this;


                // Slide-parent container and width
                var maxWidth = this.container.width() - this.moduleBar.buttonContainerRight.width() - this.moduleBar.buttonContainer.width() - 9;
                var el = $("<li/>").addClass("w-carousel-modulebar-slides").prependTo(this.moduleBar.buttonContainer);

                this.mbScroller = this.moduleBar.scrolLIze({
                    el: el,
                    maxWidth: maxWidth
                });

                return false;
            },


            rebuildIndices: function() {
                var html = "";
                this.$indices.empty();
                for (var i = 0, len = this.data.slides.length; i < len; i++) {
                    this.$indices.append('<li><a href="#" class="w-carousel-index"><span>' + (i + 1) + '</span></a></li>');
                }
                this.$indices.children().eq(this.activeSlide).addClass("active");
            },

            /**
            	 * Build multiple slides given slide data
            	 * @param 	slides 		Array slide data
            	 */
            buildSlides: function(slides) {
                if (typeof slides === 'object' && slides instanceof Array) {
                    for (var i = 0; i < slides.length; i++) 
                        this.addSlide(slides[i], i);
                }
            },


            /**
            	 * Adds a slide by creating a slide controller
            	 * @param 		slide 		Slide data to build with. If null, sample data will be used
            	 * @param 		i 			If integer, we assume slide is already on the page and i is index position in $list
            	 */
            addSlide: function(slide, i) {
                var n = this.slideControllers.length;

                if (n > 9) {
                    alert(translate("webs.module.carousel.error.maxSlides"));
                    return false;
                }

                var self = this, isNew = !slide, el,
                mbButton = this.moduleBar.addButton({
                    label: n + 1,
                    name: 'selectSlide',
                    value: function() {
                        return mbButton.children();
                    }
                    // allows us to call selectSlide on the proper controller
                }),
                tooltipSlideNumber = n + 1;
                var slideTooltip = translate("webs.module.carousel.slideCount", {
                    count: tooltipSlideNumber
                });
                mbAnchor = mbButton.children().data('index', n).attr('title', slideTooltip);



                this.mbScroller.append(mbButton);


                mbButton.children().addClass('w-carousel-modulebar-slide');

                if (isNew) {
                    slide = {
                        linksData: {},
                        image: {
                            //url: 'http://dl.dropbox.com/u/35437/images/my_images/img' + parseInt(Math.random()*18 + 1, 10) + '.jpg',
                            url: null,
                            fullWidth: 1969,
                            fullHeight: 1084,
                            width: 100,
                            left: 0,
                            top: 0
                        },
                        button: {
                            link: null,
                            align: "left",
                            linkInfo: {
                                linkType: "website",
                                linkUrl: null,
                                newWindow: false
                            },
                            icon: {
                                "enable": false,
                                "set": null,
                                "slug": null,
                                "position": "left",
                                "style": null
                            },
                            bgcolor: null,
                            background: ""
                        }
                    };
                    this.data.slides.push(slide);
                }

                /**
                		 * If we delete, deactivate, then undo or if we add, deactivate, then redo,
                		 * we get into the problem where the DOM node isn't ready yet, but we assume it
                		 * exists. We need to fake it by making a new element but with existing data
                		 */
                if (typeof i === 'number') 
                    el = this.$list.children().eq(i);
                if (!el || !el.length) {
                    el = $('<li>').appendTo(this.$list);
                    isNew = true;
                }


                this.slideControllers.push(SlideController.init({
                    parent: this,
                    el: el,
                    data: slide,
                    mbButton: mbButton,
                    mbAnchor: mbAnchor,
                    isNew: isNew
                }));
                this.bindSubmodules();

                // If index isn't passed in, it means a new slide has been created. Select that slide.
                if (typeof i !== 'number') {
                    this.selectSlide(n);
                    this.toggleArrows(true);
                }

                this.rebuildIndices();
            },


            /**
            	 * Removes a slide and associated controller.
            	 * @param 		i 		Integer slide index to delete. If not passed, current slide will be deleted.
            	 * @return 		true if successful
            	 */
            deleteSlide: function(i) {
                var self = this, sc = this.slideControllers, tmp;

                if (sc.length === 1) {
                    alert(translate("webs.module.carousel.error.noSlides"));
                    return false;
                }

                if (typeof i !== 'number') 
                    i = this.activeSlide;
                if (sc[i]) {
                    // selects a previous slide, then delete the current
                    this.selectSlide(i - 1, function() {
                        tmp = sc.splice(i, sc.length - i);
                        tmp.shift().destroy();
                        self.slideControllers = sc.concat(tmp);

                        sc = self.data.slides;
                        tmp = sc.splice(i, sc.length - i);
                        tmp.shift();
                        self.data.slides = sc.concat(tmp);

                        // Errreday I'm unshufflin' [slide index]
                        self.$buttonContainer.find('.w-carousel-modulebar-slide').each(function(i, btn) {
                            var tooltipSlideNumber = i + 1;
                            var slideTooltip = translate("webs.module.carousel.slideCount", {
                                count: tooltipSlideNumber
                            });
                            $(btn).data('index', i).attr('title', slideTooltip).html('<span>' + (i + 1) + '</span>');
                        });

                        // Fix active index
                        self.activeSlide = i === 0 ? self.slideControllers.length - 1 : i - 1;
                        self.mbScroller.check(function() {
                            self.mbScroller.select(self.activeSlide);
                        });

                        if (self.slideControllers.length === 1) {
                            self.toggleArrows(0);
                        } else {
                            self.toggleArrows(1);
                        }
                        self.bindSubmodules();

                        self.rebuildIndices();
                    });
                    return true;
                }
                return false;
            },


            /**
            	 * Selects (and activates) a slide. Will wrap around if index is out of bounds.
            	 * @param 		i 		Integer index of slide to activate
            	 * @param 		fn 		Callback since transitioning is happening, we may want to stall things like delete
            	 */
            selectSlide: function(i, fn) {
                if (i===-1) 
                    i += this.slideControllers.length;
                else if (this.slideControllers.length === i) 
                    i = 0;

                if (this.activeSlide === i) {
                    // Same index. Attempt to activate anyway as it may be from an init (module focus) call
                    this.slideControllers[i].activate();
                    return false;
                }
                if (this.transitioning) 
                    return false;

                // If there's a callback, we're probably coming from a delete or something that may
                // change data during animation. Callback is responsible for calling select method on mbScroller.
                if (typeof fn !== "function") 
                    this.mbScroller.select(i);

                var self = this,
                oldEl = this.slideControllers[this.activeSlide].el.addClass('outgoing'),
                newEl = this.slideControllers[i].el.addClass('incoming'),
                oldi = this.activeSlide,
                transition = webs.transitions[this.data.transition] || webs.transitions.none,
                invertTransition = (i < oldi) && !(i === 0 && oldi === this.slideControllers.length - 1) || 
                (i === this.slideControllers.length - 1 && oldi === 0);

                this.$carousel.addClass("w-carousel-transitioning");

                this.activeSlide = i;
                this.transitioning = true;

                transition(oldEl, newEl, this.data.duration, invertTransition, function() {
                    self.transitioning = false;
                    self.slideControllers[oldi].deactivate();
                    self.slideControllers[i].activate();
                    self.$carousel.removeClass("w-carousel-transitioning");

                    self.$indices.children().eq(oldi).removeClass("active");
                    self.$indices.children().eq(i).addClass("active");

                    if (typeof fn === 'function') 
                        fn();
                });

                self.el.trigger('slideSelected', self);
            },
            previousSlide: function() {
                this.selectSlide(this.activeSlide - 1);
                return false;
            },
            nextSlide: function() {
                this.selectSlide(this.activeSlide + 1);
                return false;
            },

            toggleArrows: function(on) {
                if (on) {
                    this.$carousel.removeClass('hide_arrows');
                } else {
                    this.$carousel.addClass('hide_arrows');
                }
            },

            /**
            	 * Module data setters
            	 */
            setAutoplay: function(on) {
                this.data.autoplay = on;
            },
            setDuration: function(speed) {
                this.data.speed = speed;
            },
            setTransition: function(transition) {
                this.data.transition = transition;
            },
            setCrop: function(height) {
                this.data.height = height;
                this.$carousel.height(height);
            },

            toggleTitle: function(on) {
                this.data.hideTitle = !on;
                if (!on) 
                    this.$carousel.addClass('hide_title');
                else 
                    this.$carousel.removeClass('hide_title');
                this.toggleDescription();
            },
            toggleText: function(on) {
                this.data.hideText = !on;
                if (!on) 
                    this.$carousel.addClass('hide_text');
                else 
                    this.$carousel.removeClass('hide_text');
                this.toggleDescription();
            },
            toggleButton: function(on) {
                this.data.hideButton = !on;
                if (!on) 
                    this.$carousel.addClass('hide_button');
                else 
                    this.$carousel.removeClass('hide_button');
                this.toggleDescription();
            },
            toggleDescription: function() {
                var data = this.data;
                data.hideDescription = data.hideTitle && data.hideText && data.hideButton;
                if (data.hideDescription) 
                    this.$carousel.addClass('hide_description');
                else 
                    this.$carousel.removeClass('hide_description');
            },

            getMinWidth: function() {
                return 200;
            }

        });


    })(mod, ext);

    return ModuleClassLoader.register('carousel', mod, ext);
});




// mobile_text Module (custom)

define('module/mobile_text', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader'],
function($, ModuleClassLoader) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="webs-mobile webs-mobile-text no-icon">\
        {{{mobileText}}}\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "mobileText": "Module text placeholder"
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS
        module.oneLoaded = function() {
            var self = this;
            var mobile_text = this.data.mobileText.replace(/\n/g, "<br />");
            if (mobile_text && mobile_text.length >= 170) {
                this.$(".webs-mobile-text").html(mobile_text.substring(0, 170) + ' ...').append('<span class="mobile-showmore">Show More</span>');
                this.$(".mobile-showmore").click(function() {
                    self.$(".webs-mobile-text").html(mobile_text);
                });
            } else {
                this.$(".webs-mobile-text").html(mobile_text);
            }
        };


        // Edit JS
        /**
        * Initialize edit mode
        */
        module.edit = function(event) {};

        module.initTooltip = function() {};

        module.onDeactivate = function() {};

        module.safeOneLoaded = function() {};

    })(mod, ext);

    return ModuleClassLoader.register('mobile_text', mod, ext);
});




// spacer Module (custom)

define('module/spacer', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.spacer'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="webs-spacer" style="height:{{height}}px; line-height: {{height}}px; ">\
        <span class="spacer_text">{{#l}}webs.module.spacer.drag{{/l}}</span>\
        <div class="webs-spacer-height"></div>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "height": 30
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS


        // Edit JS
        module.edit = function() {
            var $el = this.el.find('.webs-spacer'),
            $heightEl = $el.find('.webs-spacer-height'),
            self = this;

            $el.cropper({
                reference: $el,
                target: $heightEl,
                sides: ['bottom'],
                onMove: function(side, newWidth, newHeight) {
                    self.data.height = newHeight;
                    $el.css('line-height', newHeight + 'px');
                }
            });
        };

        module.getData = function() {
            return this.data;
        };

    })(mod, ext);

    return ModuleClassLoader.register('spacer', mod, ext);
});




// button Module (custom)

define('module/button', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.button'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="w-button-container webs-align-{{align}}">\
        <div class="w-button-align webs-align-{{align}}">\
        {{#if labelFor}}\
        <label for="{{labelFor}}">\
        {{/if}}\
        <a class="w-button{{#if icon}}{{#if icon.enable}} w-button-icon w-button-icon-set-{{icon.set}} w-button-icon-pos-{{icon.position}}{{/if}}{{/if}}" {{#if bgcolor}} style="background-color: {{bgcolor}}"{{/if}}>\
        <div class="w-button-inner">\
        <div class="w-button-text-wrapper">\
        <div class="w-button-text" contenteditable="true" data-placeholder="{{#l}}webs.module.button.clicktoedit{{/l}}" spellcheck="true">{{#if html}}{{{html}}}{{else}}{{#l}}webs.module.button.clicktoedit{{/l}}{{/if}}</div>\
        </div>\
        {{#if icon}}{{#if icon.enable}}\
        <div class="w-button-icon-wrapper">\
        <div class="w-button-icon-icon">{{renderIcon icon}}</div>\
        </div>\
        {{/if}}{{/if}}\
        </div>\
        </a>\
        {{#if labelFor}}\
        </label>\
        {{/if}}\
        </div>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "html": null,
            "link": null,
            "bgcolor": null,
            "align": "left",
            "linkInfo": {
                "linkType": "website",
                "linkUrl": null,
                "newWindow": false
            },
            "icon": {
                "enable": false,
                "set": null,
                "slug": null,
                "position": "left",
                "style": null
            }
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS


        // Edit JS
        /**
        * Initialize edit mode
        */
        module.onActivate = function(event) {};

        module.attachControls = function(event) {
            var $ = bldr.pageController.$,
            data = this.data,
            self = this;
            this.controls = bldr.controls.create('button', {
                el: this.el,
                data: data,
                enableIcon: true,
                render: function() {
                    self.render();
                }
            });
            this.controls.show();
            var editFrame = top.$("iframe#website")[0],
            editWindow = editFrame.contentWindow || editFrame.contentDocument.parentWindow;
            this.rangy = editWindow.rangy;
        };

        module.beforeUndo = function() {
            if (this.controls && this.controls.wizzy) {
                this.sessionRange = this.controls.wizzy.getRange();
            }
        };

        module.afterUndo = function() {
            if (this.controls && this.controls.wizzy) {
                this.controls.wizzy.setElement(this.el.find('.w-button-text'));
            }
            var range = this.sessionRange;
            if (!range || !range.isValid()) {
                range = this.rangy.createRange();
                range.selectNodeContents(this.el.find('.w-button-text')[0]);
                range.collapse();
            }
            this.rangy.getSelection().addRange(range);
        };

        module.onDeactivate = function() {
            if (this.controls) {
                this.data = this.controls.getData();
                this.controls.destroy();
            }
        };

        module.getData = function() {
            if (this.controls) 
                this.data = this.controls.getData();
            return this.data;
        };


    })(mod, ext);

    return ModuleClassLoader.register('button', mod, ext);
});




// social-links Module (custom)

define('module/social-links', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.social-links'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="w-social_links">\
        {{#if links}}\
        <div class="w-social-links_resizer" style="{{#if width}}width: {{width}}px;{{/if}}">\
        {{#if links.facebook}}\
        <a href="{{links.facebook}}" class="w-social_link w-social_link-facebook"></a>\
        {{/if}}\
        {{#if links.twitter}}\
        <a href="{{links.twitter}}" class="w-social_link w-social_link-twitter"></a>\
        {{/if}}\
        {{#if links.linkedin}}\
        <a href="{{links.linkedin}}" class="w-social_link w-social_link-linkedin"></a>\
        {{/if}}\
        {{#if links.youtube}}\
        <a href="{{links.youtube}}" class="w-social_link w-social_link-youtube"></a>\
        {{/if}}\
        {{#if links.rss}}\
        <a href="{{links.rss}}" class="w-social_link w-social_link-rss"></a>\
        {{/if}}\
        {{#if links.tumblr}}\
        <a href="{{links.tumblr}}" class="w-social_link w-social_link-tumblr"></a>\
        {{/if}}\
        {{#if links.blogger}}\
        <a href="{{links.blogger}}" class="w-social_link w-social_link-blogger"></a>\
        {{/if}}\
        {{#if links.flickr}}\
        <a href="{{links.flickr}}" class="w-social_link w-social_link-flickr"></a>\
        {{/if}}\
        {{#if links.vimeo}}\
        <a href="{{links.vimeo}}" class="w-social_link w-social_link-vimeo"></a>\
        {{/if}}\
        {{#if links.lastfm}}\
        <a href="{{links.lastfm}}" class="w-social_link w-social_link-lastfm"></a>\
        {{/if}}\
        {{#if links.pinterest}}\
        <a href="{{links.pinterest}}" class="w-social_link w-social_link-pinterest"></a>\
        {{/if}}\
        {{#if links.instagram}}\
        <a href="{{links.instagram}}" class="w-social_link w-social_link-instagram"></a>\
        {{/if}}\
        </div>\
        {{else}}\
        <div class="bldr-placeholder-element">\
        <div class="instructions">\
        {{#l}}webs.module.social-links.placeholderInstructions{{/l}}\
        </div>\
        </div>\
        {{/if}}\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "links": null
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "base": {
                "global": {
                    "css": "view.less"
                },
                "slug": "base"
            },
            "style1": {
                "inherit": "base",
                "default": true,
                "slug": "style1"
            },
            "style2": {
                "inherit": "base",
                "slug": "style2"
            },
            "style3": {
                "inherit": "base",
                "slug": "style3"
            },
            "style4": {
                "inherit": "base",
                "slug": "style4"
            },
            "style5": {
                "inherit": "base",
                "slug": "style5"
            },
            "style6": {
                "inherit": "base",
                "slug": "style6"
            },
            "style7": {
                "inherit": "base",
                "slug": "style7"
            },
            "style8": {
                "inherit": "base",
                "slug": "style8"
            }
        };
        if (!extend.styles['base']['global']) {
            extend.styles['base']['global'] = {};
        }
        extend.styles['base']['global']['js'] = null;
        if (!extend.styles['style1']['global']) {
            extend.styles['style1']['global'] = {};
        }
        extend.styles['style1']['global']['js'] = null;
        if (!extend.styles['style2']['global']) {
            extend.styles['style2']['global'] = {};
        }
        extend.styles['style2']['global']['js'] = null;
        if (!extend.styles['style3']['global']) {
            extend.styles['style3']['global'] = {};
        }
        extend.styles['style3']['global']['js'] = null;
        if (!extend.styles['style4']['global']) {
            extend.styles['style4']['global'] = {};
        }
        extend.styles['style4']['global']['js'] = null;
        if (!extend.styles['style5']['global']) {
            extend.styles['style5']['global'] = {};
        }
        extend.styles['style5']['global']['js'] = null;
        if (!extend.styles['style6']['global']) {
            extend.styles['style6']['global'] = {};
        }
        extend.styles['style6']['global']['js'] = null;
        if (!extend.styles['style7']['global']) {
            extend.styles['style7']['global'] = {};
        }
        extend.styles['style7']['global']['js'] = null;
        if (!extend.styles['style8']['global']) {
            extend.styles['style8']['global'] = {};
        }
        extend.styles['style8']['global']['js'] = null;
        extend.defaultStyle = extend.styles['style1'];

        // Register Toolbars
        bldr.toolbar.register('social-links', {
            "icon": "social-links",
            "items": [{
                "type": "button",
                "style": "blue",
                "label": "Manage Social",
                "name": "links",
                "dialog": {
                    "url": "/s/modules/social-links/v1.18/social-links.dialog.html",
                    "localizeHeading": "webs.module.social-links.popoverTitle",
                    "width": 440,
                    "height": 545
                }
            }, {
                "type": "radio",
                "title": "",
                "label": "Icon Styles",
                "menuLabel": "Social Icon Styles",
                "name": "style",
                "width": 202,
                "menuPosition": "center",
                "icon": "star",
                "menuClassNames": "w-social-icons-menu-radio",
                "menuItemClassNames": "w-social-icons-menu-radio-item",
                "options": [{
                    "icon": "/s/modules/social-links/v1.18/icons/sample01.png",
                    "value": "style1",
                    "top": "6px"
                }, {
                    "icon": "/s/modules/social-links/v1.18/icons/sample02.png",
                    "value": "style2",
                    "top": "12px"
                }, {
                    "icon": "/s/modules/social-links/v1.18/icons/sample03.png",
                    "value": "style3",
                    "top": "2px"
                }, {
                    "icon": "/s/modules/social-links/v1.18/icons/sample04.png",
                    "value": "style4",
                    "top": "6px"
                }, {
                    "icon": "/s/modules/social-links/v1.18/icons/sample05.png",
                    "value": "style5",
                    "top": "5px"
                }, {
                    "icon": "/s/modules/social-links/v1.18/icons/sample06.png",
                    "value": "style6",
                    "top": "12px"
                }, {
                    "icon": "/s/modules/social-links/v1.18/icons/sample07.png",
                    "value": "style7",
                    "top": "9px"
                }, {
                    "icon": "/s/modules/social-links/v1.18/icons/sample08.png",
                    "value": "style8",
                    "top": "9px"
                }
                ]
            }
            ]
        });

        // View JS


        // Edit JS
        module.onActivate = function() {};
        module.onDeactivate = function() {};

        module.showSettingsDialog = function() {
            this.toolbar.click("links");
        };

        module.attachControls = function() {
            var self = this,
            $ = bldr.pageController.$, data = self.data, container = self.el;

            self.toolbar = bldr.toolbar.create('social-links');
            self.toolbar.show({
                links: this.data.links,
                style: this.getCurrentStyle().slug
            });

            self.toolbar.addListener({
                links: function(data) {
                    self.data.links = data;
                    self.render();
                },
                style: function(data) {
                    self.setStyle(data);
                    self.render();
                }
            });

            this.resizer = container.find('.w-social-links_resizer').addClass("active");

            container.find("a").click(function(e) {
                e.preventDefault();
                return false;
            });

            container.find(".w-social_links").bind("dblclick", function(e) {
                self.toolbar.click("links");
                e.preventDefault();
                return false;
            });

            this.sizeControl = $("<div/>").
            addClass("bldr-resize-handle bldr-resize-handle-right").
            appendTo(this.resizer).
            drag({
                drag: false
            }).
            bind("drag", function(event, x, y) {
                //self.data.height = Math.max(x, self.minHeight);
                self.data.width = Math.min(Math.max(y, self.minWidth), self.maxWidth);
                self.resizer /*.height(self.data.height)*/
                .width(self.data.width);
            }).bind("dragEnd", function() {
});

        };

        module.minWidth = 32;
        module.maxWidth = 100;

        module.fit = function(width, persist) {
            this.maxWidth = width;
            if (this.maxWidth < this.data.width) {
                this.el.find(".w-social-links_resizer").width(this.maxWidth);
                if (persist) {
                    this.data.width = this.maxWidth;
                    this.dirty();
                }
            }
        };

    })(mod, ext);

    return ModuleClassLoader.register('social-links', mod, ext);
});




// contact_form Module (custom)

define('module/contact_form', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.contact_form'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };

        extend.settingsDialog = {
            "url": "/s/modules/contact_form/v1.33/contact_form.dialog.html",
            "localizeHeading": "webs.module.contact_form.dialogHeader",
            "width": 750,
            "height": 510
        };


        // Module HTML Template
        extend.template = '<div class="w-contact_form webs-composite-module">\
        <div class="w-contact_form-container">\
        <form action="/s/site_module/{{encodeURIComponent _site.id}}/contact_form/{{encodeURIComponent _id}}" method="GET" novalidate>\
        <input type="text" name="fw_human" style="display: none;"/>\
        {{#case layout "left"}}\
        <table class="clearfix w-contact_form-ul w-contact_form-label-{{layout}}">\
        {{#each fields}}\
        <tr class="w-contact_form-li w-contact_form-{{type}}">\
        <td class="w-contact_form-label">\
        <label class="{{#if required}}required{{/if}}" for="w-form-{{../../_id}}-{{_i_}}">{{label}}</label>\
        </td>\
        <td class="w-contact_form-input">\
        {{#case type "text"}}\
        <input class="w-contact_form-field" type="text" id="w-form-{{../../_id}}-{{_i_}}" name="w-form-{{../../_id}}-{{_i_}}" />\
        {{/case}}\
        {{#case type "email"}}\
        <input class="w-contact_form-field" type="email" id="w-form-{{../../_id}}-{{_i_}}" name="w-form-{{../../_id}}-{{_i_}}" />\
        {{/case}}\
        {{#case type "textarea"}}\
        <textarea class="w-contact_form-field" id="w-form-{{../../_id}}-{{_i_}}" name="w-form-{{../../_id}}-{{_i_}}"></textarea>\
        {{/case}}\
        {{#case type "checkbox"}}\
        <ul class="w-contact_form-options">\
        {{#each_object options}}\
        <li>\
        <input type="checkbox" name="w-form-{{../../../_id}}-{{../_i_}}" value="{{val}}" id="w-form-{{../../../_id}}-{{../_i_}}-opt_{{_i_}}" />\
        <label for="w-form-{{../../../_id}}-{{../_i_}}-opt_{{_i_}}">{{val}}</label>\
        </li>\
        {{/each_object}}\
        </ul>\
        {{/case}}\
        {{#case type "radio"}}\
        <ul class="w-contact_form-options">\
        {{#each_object options}}\
        <li>\
        <input type="radio" name="w-form-{{../../../_id}}-{{../_i_}}" value="{{val}}" id="w-form-{{../../../_id}}-{{../_i_}}-opt_{{_i_}}"/>\
        <label for="w-form-{{../../../_id}}-{{../_i_}}-opt_{{_i_}}">{{val}}</label>\
        </li>\
        {{/each_object}}\
        </ul>\
        {{/case}}\
        {{#case type "dropdown"}}\
        <select class="w-contact_form-select" name="w-form-{{../../_id}}-{{_i_}}" id="w-form-{{../../_id}}-{{_i_}}">\
        {{#each_object options}}\
        <option>{{val}}</option>\
        {{/each_object}}\
        </select>\
        {{/case}}\
        </td>\
        </tr>\
        {{/each}}\
        <tr class="w-contact_form-li">\
        <td>&nbsp;</td>\
        <td class="w-contact_form-input">\
        <input type="submit" id="{{button.labelFor}}" class="w-contact_form-submit"/>\
        {{submodule "button" button}}\
        </td>\
        </tr>\
        </table>\
        {{else}}\
        <ul class="clearfix w-contact_form-ul w-contact_form-label-{{layout}}">\
        {{#each fields}}\
        <li class="clearfix w-contact_form-li w-contact_form-{{type}}">\
        <label class="{{#if required}}required{{/if}}" for="w-form-{{../../_id}}-{{_i_}}">{{label}}</label>\
        <div class="w-contact_form-input">\
        {{#case type "text"}}\
        <input class="w-contact_form-field" type="text" id="w-form-{{../../_id}}-{{_i_}}" name="w-form-{{../../_id}}-{{_i_}}" />\
        {{/case}}\
        {{#case type "email"}}\
        <input class="w-contact_form-field" type="email" id="w-form-{{../../_id}}-{{_i_}}" name="w-form-{{../../_id}}-{{_i_}}" />\
        {{/case}}\
        {{#case type "textarea"}}\
        <textarea class="w-contact_form-field" id="w-form-{{../../_id}}-{{_i_}}" name="w-form-{{../../_id}}-{{_i_}}"></textarea>\
        {{/case}}\
        {{#case type "checkbox"}}\
        <ul class="w-contact_form-options">\
        {{#each_object options}}\
        <li>\
        <input type="checkbox" name="w-form-{{../../../../_id}}-{{../_i_}}" value="{{val}}" id="w-form-{{../../../../_id}}-{{../_i_}}-opt_{{_i_}}" />\
        <label for="w-form-{{../../../../_id}}-{{../_i_}}-opt_{{_i_}}">{{val}}</label>\
        </li>\
        {{/each_object}}\
        </ul>\
        {{/case}}\
        {{#case type "radio"}}\
        <ul class="w-contact_form-options">\
        {{#each_object options}}\
        <li>\
        <input type="radio" name="w-form-{{../../../../_id}}-{{../_i_}}" value="{{val}}" id="w-form-{{../../../../_id}}-{{../_i_}}-opt_{{_i_}}"/>\
        <label for="w-form-{{../../../../_id}}-{{../_i_}}-opt_{{_i_}}">{{val}}</label>\
        </li>\
        {{/each_object}}\
        </ul>\
        {{/case}}\
        {{#case type "dropdown"}}\
        <select class="w-contact_form-select" name="w-form-{{../../../_id}}-{{_i_}}" id="w-form-{{../../_id}}-{{_i_}}">\
        {{#each_object options}}\
        <option>{{val}}</option>\
        {{/each_object}}\
        </select>\
        {{/case}}\
        </div>\
        </li>\
        {{/each}}\
        <li class="w-contact_form-li">\
        <input type="submit" id="{{button.labelFor}}" class="w-contact_form-submit"/>\
        {{submodule "button" button}}\
        </li>\
        </ul>\
        {{/case}}\
        </form>\
        <div class="success-message">\
        <div class="success-icon"></div>\
        {{{text_to_html confirmation}}}\
        </div>\
        <div class="error-message">\
        <div class="error-icon"></div>\
        <span class="error-string">{{#l}}webs.module.contact_form.errorSubmitting{{/l}}</span>\
        </div>\
        </div>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "untranslated": true,
            "layout": "top",
            "email": null,
            "confirmation": "webs.module.contact_form.defaultConfirmation",
            "button": {
                "labelFor": "contact_form-generic-uid-submit",
                "html": "webs.module.contact_form.defaultButtonText",
                "link": null,
                "linkInfo": {},
                "linkFrozen": true,
                "icon": {
                    "enable": false,
                    "set": null,
                    "slug": null,
                    "position": "left",
                    "style": null
                },
                "iconLeft": true,
                "iconStyle": "light",
                "bgcolor": null,
                "background": ""
            },
            "fields": [{
                "type": "text",
                "label": "webs.module.contact_form.defaultNameLabel",
                "default": "Enter name here",
                "required": true
            }, {
                "type": "email",
                "label": "webs.module.contact_form.defaultEmailLabel",
                "default": "Enter Email",
                "required": true
            }, {
                "type": "textarea",
                "label": "webs.module.contact_form.defaultMessageLabel",
                "default": "Enter Message",
                "required": true
            }
            ]
        };

        // SubModules
        extend.submodules = {
            "button": {
                "moduleType": "button"
            }
        };

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less",
                    "js": "view.js"
                },
                "default": true,
                "slug": "default"
            }
        };
        extend.styles['default']['global']['js'] = (function(module, extend) {


            /* view.js */
            module.oneLoaded = function() {
                this.parent.parent.fn.oneLoaded.call(this);
                var self = this;

                var form = self.el.find("form"),
                submitButton = self.el.find(".w-button"),
                submitLabel = submitButton.parents("label"),
                resubmit_link = self.el.find(".resubmit_form_link"),
                prohibitedLabels = [],
                tokens = [];

                // IE 8 and below doesn't like this label wrapping button to trigger form idea
                // Manually attaching form submit call to button
                submitButton.click(function() {
                    form.submit();
                    return false;
                });
                submitLabel.click(function() {
                    return false;
                });




                form.submit(function() {
                    var missingFields = [];

                    form.find(".required").each(function(index, label) {
                        label = $(label).attr("for");
                        var formItem = form.find("[name='" + label + "']"),
                        formItemContainer = formItem.closest('.w-contact_form-li');

                        if (!formItem.length)
                            formItem = form.find("#" + label);
                        if (!formItem.length) {
                            missingFields.push({
                                "labelFor": label,
                                "reason": "missing"
                            });
                            return;
                        }
                        switch (formItem[0].nodeName.toLowerCase()) {
                        case "input":
                            switch (formItem.attr("type")) {
                            case "text":
                                if (formItem.val().match(/^\s*$/) || formItemContainer.hasClass("error-item"))
                                    missingFields.push({
                                        "labelFor": label,
                                        "reason": "blank"
                                    });
                                break;
                            case "email":
                                if (!formItem.val().match(/^\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4})\s*$/))
                                    missingFields.push({
                                        "labelFor": label,
                                        "reason": "invalidEmail"
                                    });
                                break;
                            case "checkbox":
                            case "radio":
                                if (form.find("[name='" + label + "']:checked").length === 0)
                                    missingFields.push({
                                        "labelFor": label,
                                        "reason": "noCheck"
                                    });
                                break;
                            }
                            break;
                        case "textarea":
                            if (formItem.val().match(/^\s*$/) || formItemContainer.hasClass("error-item"))
                                missingFields.push({
                                    "labelFor": label,
                                    "reason": "blank"
                                });
                            break;
                        case "select":
                            // Is there any way to check this, without some placeholder option that's considered a non-answer?
                            break;
                        }
                    });
                    if (missingFields.length === 0) {
                        $.ajax({
                            url: webs.props.membersServer + form.attr("action"),
                            type: form.attr("method"),
                            data: form.serialize(),
                            dataType: "jsonp",
                            contentType: 'application/json;charset=UTF-8',
                            success: function(data) {
                                var success,
                                padding,
                                newHeight,
                                failure;

                                if (data.success) {
                                    success = self.el.find(".success-message");
                                    padding = (form.height() - success.height()) / 2;
                                    newHeight = form.height();
                                    if (newHeight < success.height()) 
                                        newHeight = success.height();
                                    if (padding < 0) 
                                        padding = 0;
                                    success.css({
                                        "padding-top": padding + "px",
                                        "padding-bottom": padding + "px"
                                    });
                                    self.el.find(".w-contact_form-container").height(form.height()).addClass("success");
                                } else {
                                    failure = self.el.find(".error-message");
                                    string = failure.find(".error-string");
                                    if (data.reason === "label_error") {
                                        string.html(translate("webs.module.contact_form.invalidForm"));
                                    }
                                    padding = (form.height() - failure.height()) / 2;
                                    newHeight = form.height();
                                    if (newHeight < failure.height()) 
                                        newHeight = failure.height();
                                    if (padding < 0) 
                                        padding = 0;
                                    failure.css({
                                        "padding-top": padding + "px",
                                        "padding-bottom": padding + "px"
                                    });
                                    self.el.find(".w-contact_form-container").height(newHeight).addClass("failure");
                                }
                            }
                        });
                    } else {
                        form.addClass("error");

                        $.each(missingFields, function(i, missing) {
                            var label = self.el.find("label[for='" + missing.labelFor + "']"),
                            labelContainer = label.closest('.w-contact_form-li'),
                            input;

                            labelContainer.addClass("error-item");

                            switch (self.el.find("[name='" + missing.labelFor + "']").attr("type")) {
                            case "checkbox":
                            case "radio":
                                self.el.find("[name='" + missing.labelFor + "']").bind("focus click change", function() {
                                    labelContainer.removeClass("error-item");
                                    if (!form.find(".error-item").length) 
                                        form.removeClass("error");
                                });
                                break;
                            default:
                                input = form.find("#" + missing.labelFor);
                                var inputValue = "",
                                inputLabel = form.find("label[for='" + missing.labelFor + "']").text();
                                if (missing.reason === "invalidEmail") {
                                    inputValue = translate('webs.module.contact_form.error.invalidEmail', {
                                        label: inputLabel
                                    });
                                } else {
                                    inputValue = translate('webs.module.contact_form.error.blank', {
                                        label: inputLabel
                                    });
                                }
                                input.val(inputValue);
                                input.bind("focus click keydown cut paste change", function() {
                                    input.unbind("focus click keydown cut paste change");
                                    input.val("");
                                    labelContainer.removeClass("error-item");
                                    if (!form.find(".error-item").length) 
                                        form.removeClass("error");
                                });
                            }
                        });
                    }
                    return false;
                });
                resubmit_link.click(function() {
                    self.el.find(".w-contact_form-container").removeClass("success").removeClass("failure");
                });
            };
            return module;
        });
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars
        bldr.toolbar.register('contact_form', {
            "icon": "contact_form",
            "items": [{
                "type": "button",
                "style": "blue",
                "label": "Edit Form",
                "name": "dialog",
                "dialog": {
                    "url": "/s/modules/contact_form/v1.33/contact_form.dialog.html",
                    "localizeHeading": "webs.module.contact_form.dialogHeader",
                    "width": 750,
                    "height": 510
                }
            }
            ]
        });

        // View JS


        // Edit JS
        module.safeOneLoaded = function() {
            this.translateDefaults();
            var self = this;
            setTimeout(function() {
                self.dirty();
            }, 0);
        };

        module.onActivate = function(event) {
            this.ensureId();
        };

        module.translateDefaults = function() {
            if (this.data.untranslated) {
                this.data.untranslated = false;
                this.data.confirmation = translate(this.data.confirmation);
                $.each(this.data.fields, function(i, field) {
                    field.label = translate(field.label);
                });
                this.data.button.html = translate(this.data.button.html);
            }
        };

        module.attachCompositeControls = function(event) {
            var self = this;
            var $container = self.el;

            var toolbar = bldr.toolbar.create('contact_form');
            toolbar.feedback = {
                ref: $container 
            };
            toolbar.show({
                'dialog': self.data 
            });
            toolbar.addListener({
                'dialog': function(data) {
                    self.data = data;
                    self.render();
                }
            });

            self.editMode = 'form';
            $container.bind('click', function() {
                if (self.editMode != 'form') {
                    toolbar.show({
                        'dialog': self.data
                    });
                    self.deactivateSubmodules();
                    self.editMode = 'form';
                    return false;
                }
                return false;
            });

            $container.unbind("dblclick.module");
            $container.bind("dblclick.module", function(e) {
                if (!$(e.srcElement || e.target).parents(".w-button").length)
                    toolbar.click('dialog');
            });

            this.el.find(".webs-submodule-button").bind("click", function(e) {
                if (self.editMode != "button") {
                    self.activateSubmodule("button");
                    self.editMode = "button";
                    e.stopPropagation();
                    return false;
                }
                return false;
            });
            $container.find("input,select,textarea").each(function(i, field) {
                field.disabled = true;
            });
        };

        module.onDeactivate = function() {};

        module.ensureId = function() {
            if (!this.data._uid)
                this.data._uid = Spine.guid();
        };

        module.getData = function() {
            this.ensureId();
            if (this.submoduleInstances)
                this.data.button = this.submoduleInstances.button.getData();
            this.data.button.labelFor = "contact_form-" + this.data._uid + "-submit";
            return this.data;
        };

        module.getMinWidth = function() {
            return 200;
        };


    })(mod, ext);

    return ModuleClassLoader.register('contact_form', mod, ext);
});




// video Module (custom)

define('module/video', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.video'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '{{#wrapInFrame}}\
        \
        {{#if videoType}}\
        \
        {{#case videoType "youtube"}}\
        <div class="webs-video-wrapper {{#if frame}}webs-hasframe{{/if}} webs-video-youtube webs-align-{{align}}" style="width: {{width}}px; height: {{height}}px;">\
        <div class="webs-video-crop">\
        <img class="webs-youtube-preview" src="{{#id}}http://img.youtube.com/vi/{{.}}/hqdefault.jpg{{^}}http://static.websimages.com/static/projects/finch/images/placeholder_video.jpg{{/id}}"/>\
        </div>\
        <div class="webs-video-control-1">\
        <div class="webs-video-control-2"></div>\
        </div>\
        </div>\
        {{/case}}\
        \
        {{#case videoType "browse"}}\
        <div class="webs-video-wrapper {{#if frame}}webs-hasframe{{/if}} webs-video-vidly webs-align-{{align}}" style="width: {{width}}px; height: {{height}}px;">\
        <div class="webs-video-complete">\
        <div class="webs-video-crop">\
        <img class="webs-vidly-preview" src="{{#id}}http://cf.cdn.vid.ly/{{.}}/poster.jpg{{^}}http://static.websimages.com/static/projects/finch/images/placeholder_video.jpg{{/id}}"/>\
        </div>\
        <div class="webs-video-control-1">\
        <div class="webs-video-control-2"></div>\
        </div>\
        </div>\
        <div class="webs-video-uploading bldr-placeholder-loading" data-placeholder="{{#l}}webs.module.video.videoIsCurrentlyBeingProcessed{{/l}}" style="width: 100%; height: 100%;"></div>\
        <div class="webs-video-error bldr-placeholder-element" data-placeholder="{{#l}}webs.module.video.problemUploadingVideo{{/l}}" style="width: 100%; height: 100%;"></div>\
        </div>\
        {{/case}}\
        \
        {{#case videoType "embed"}}\
        <div class="webs-video-wrapper {{#if frame}}webs-hasframe{{/if}} webs-video-embed webs-align-{{align}}" style="width: {{embedWidth}}px; height: {{embedHeight}}px;">\
        <div class="webs-video-crop">\
        <div class="bldr-placeholder-element" data-placeholder="{{#l}}webs.module.video.embedCode.image_placeholder{{/l}}"></div>\
        </div>\
        </div>\
        {{/case}}\
        \
        {{else}}\
        \
        <div class="webs-video-wrapper {{#if frame}}webs-hasframe{{/if}} webs-align-{{align}}" style="width: {{width}}px; height: {{height}}px;">\
        <div class="webs-video-crop {{#unless videoType}}bldr-placeholder-element{{/unless}}" data-placeholder="{{#l}}webs.module.video.doubleClickToChangeVideo{{/l}}">\
        <img src="http://static.websimages.com/static/projects/finch/images/placeholder_video.jpg"/>\
        </div>\
        <div class="webs-video-control-1">\
        <div class="webs-video-control-2"></div>\
        </div>\
        </div>\
        \
        {{/if}}\
        \
        {{/wrapInFrame}}\
        ';

        // Module Default Data
        extend.defaultData = {
            "url": null,
            "id": null,
            "videoType": null,
            "maxWidth": 320,
            "minWidth": 180,
            "ratio": 0.75,
            "width": 320,
            "height": 240,
            "align": "center",
            "frame": false,
            "frameColor": "ffffff"
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars
        bldr.toolbar.register('video', {
            "icon": "video",
            "items": [{
                "label": "Change Video",
                "name": "changeVideo",
                "style": "blue",
                "dialog": {
                    "url": "/s/modules/video/v1.23/video.dialog.html",
                    "localizeHeading": "webs.module.video.dialogHeader",
                    "width": 590,
                    "height": 366
                }
            }, {
                "icon": "border",
                "type": "framesets",
                "label": "Frame",
                "name": "frame",
                "width": 170
            }, {
                "type": "button-group",
                "sticky": true,
                "distinct": true,
                "items": [{
                    "label": "Left",
                    "icon": "align_left",
                    "name": "align",
                    "value": "left"
                }, {
                    "label": "Center",
                    "icon": "align_center",
                    "name": "align",
                    "value": "center"
                }, {
                    "label": "Right",
                    "icon": "align_right",
                    "name": "align",
                    "value": "right"
                }
                ]
            }
            ]
        });

        // View JS
        if (webs && webs.theme && webs.theme.mobile) {
            module.oneLoaded = function() {
                var wrap = this.el.find('.webs-video-wrapper'),
                iframe = this.el.find('iframe'),
                wrapWidth = wrap.width(),
                wrapHeight = wrap.height(),
                contentWidth = this.el.hasClass('webs-container') ? this.el.width() : this.el.parents('.webs-container').eq(0).width(),
                widthDelta = wrapWidth - contentWidth,
                newWidth = wrapWidth - widthDelta,
                ratio = newWidth / wrapWidth,
                newHeight = wrapHeight * ratio;

                if (ratio < 1) {
                    wrap.css({
                        width: newWidth,
                        height: newHeight
                    });
                    iframe.attr({
                        width: newWidth,
                        height: newHeight
                    });
                }
                this.el.addClass('mobile-visible');
            }
        }

        // Edit JS
        /**
        * Initialize edit mode
        */

        module.frameWidth = function() {
            var frame = this.el.find(".webs-frame"),
            inner = this.el.find(".webs-video-wrapper");

            if (frame.length === 0) 
                return 0;
            return frame.outerWidth() - inner.outerWidth();
        };

        module.edit = function() {
            var self = this,
            $container = this.el,
            wrapper = $container.find('.webs-video-wrapper'),
            data = this.data,

            _width,
            _height,

            clearBorder = function() {
                wrapper.removeClass (function (index, css) {
                    return (css.match (/\bw-frame_\S+/g) || []).join(' ');
                });
            };

            this.drawControls();

            self.toolbar = bldr.toolbar.create('video');
            var toolbarBindings = {
                frameColor: function(param, name) {
                    if ('change' === param.event) {
                        self.data.frameColor = param.hex;
                        wrapper.css("border-color", "#" + param.hex);
                    }
                },
                setFrame: function(slug) {
                    var frameWidth = self.frameWidth();
                    if (typeof slug !== 'string') {
                        if (data.frame) 
                            delete data.frame;
                    } else {
                        data.frame = slug;
                        data.frameColor = data.frameColor || webs.theme.variables['accent_color'] || null;
                    }
                    self.render();

                    if (data.videoType !== "embed") {
                        var newFrameWidth = self.frameWidth();
                        data.width -= newFrameWidth - frameWidth;
                        data.maxWidth -= newFrameWidth - frameWidth;
                        data.height = parseInt(data.width * data.ratio, 10);
                        $container.find(".webs-video-wrapper").width(data.width).height(data.height);
                    }
                },
                frameRemove: function() {
                    self.data.frame = false;
                    clearBorder();
                },

                changeVideo: function(obj, name) {
                    if (!obj || !obj.videoType) 
                        return;

                    data.videoType = obj.videoType;
                    data.id = obj.id;
                    if (obj.embedWidth) {
                        data.embedWidth = obj.embedWidth;
                    } else {
                        delete data.embedWidth;
                    }
                    if (obj.embedHeight) {
                        data.embedHeight = obj.embedHeight;
                    } else {
                        delete data.embedHeight;
                    }
                    self.render();
                },

                align: function(val, name) {
                    $container.find(data.frame ? '.webs-frame' : '.webs-video-wrapper').
                    removeClass('webs-align-' + data.align).
                    addClass('webs-align-' + val);
                    data.align = val;
                }
            };
            self.toolbar.bind(toolbarBindings);

            self.toolbar.show({
                'align': data.align,
                'changeVideo': {
                    url: this.data.url,
                    videoType: this.data.videoType,
                    id: this.data.id 
                },
                'frame': this.data.frame
            });

            var proxiedRedraw = this.proxy(this.redraw);
            var unbindHandler = function() {
                this.unbind("rendered", proxiedRedraw);
                this.unbind("blurred", unbindHandler);
            };

            this.bind("rendered", proxiedRedraw);
            this.bind("blurred", unbindHandler);

            this.attachListeners();
        };

        module.redraw = function() {
            this.attachListeners();
            this.drawControls();
        };

        module.attachListeners = function() {

            var self = this;
            var wrapper = this.el.find('.webs-video-wrapper');

            this.toolbar.setFeedback(wrapper);
            wrapper.dblclick(function() {
                self.toolbar.click('changeVideo');
            });

        };

        var $sideTT = null,
        $bodyTT = null;

        module.drawControls = function() {

            var self = this,
            $container = this.el,
            wrapper = $container.find('.webs-video-wrapper'),
            data = this.data,

            _width,
            _height,

            clearBorder = function() {
                wrapper.removeClass (function (index, css) {
                    return (css.match (/\bw-frame_\S+/g) || []).join(' ');
                });
            };

            if (!$sideTT)
                $sideTT = bldr.pageController.$.tooltip({
                    style: "warn",
                    content: translate('webs.module.video.warning.sidebarMaxWidth')
                });
            if (!$bodyTT)
                $bodyTT = bldr.pageController.$.tooltip({
                    style: "warn",
                    content: translate('webs.module.video.warning.containerMaxWidth')
                });

            if (this.data.videoType && this.data.videoType != "embed") {
                var frameWidth = self.frameWidth(),
                maxWidth = $container.width() - frameWidth,
                minWidth = data.minWidth - frameWidth;

                if (minWidth > maxWidth) 
                    minWidth = maxWidth;

                wrapper.resizer({
                    reference: wrapper,
                    corners: ['left', 'right'],
                    centered: data.align === 'center',
                    maxWidth: maxWidth,
                    minWidth: minWidth,
                    onMove: function(corner, newWidth, newHeight) {
                        _width = newWidth;
                        _height = newHeight;
                        if (newWidth == maxWidth) {
                            if (self.el.parents(".webs-sidebar").length) {
                                $sideTT.addClass("active");
                            } else {
                                $bodyTT.addClass("active");
                            }
                        } else {
                            $sideTT.removeClass("active");
                            $bodyTT.removeClass("active");
                        }
                    },
                    onEnd: function() {
                        $sideTT.removeClass("active");
                        $bodyTT.removeClass("active");
                        data.maxWidth = _width;
                        data.width = _width;
                        data.height = _height;
                    }
                });
            }

        };

        module.fit = function(width, persist) {
            var $container = this.el,
            wrapper = $container.find('.webs-video-wrapper'),
            data = this.data,
            newWidth = data.width,
            newHeight = data.height,
            maxInnerWidth;

            if (data.videoType && data.videoType == "embed") 
                return;

            maxInnerWidth = width - this.frameWidth();

            if (maxInnerWidth < data.width) {
                // resize video to fit within smaller maximum width
                newWidth = maxInnerWidth;
                newHeight = parseInt(data.height / data.width * newWidth, 10);
            } else if (data.width < maxInnerWidth) {
                // resize video larger if it was previously resized smaller by fit
                newWidth = Math.min(data.maxWidth, maxInnerWidth);
                newHeight = parseInt(newWidth * data.ratio, 10);
            } else {
                return false;
            }

            wrapper.width(newWidth).height(newHeight);

            if (persist) {
                data.width = newWidth;
                data.height = newHeight;
                this.dirty();
            }
        };

        module.getMinWidth = function() {
            var videoWidth = 100;
            if (this.data.embedWidth)
                videoWidth = this.data.embedWidth;
            return videoWidth + 20 + this.frameWidth();
        };

        module.getData = function() {
            return this.data;
        };

        module.checkStatus = function () {
            var self = this;
            $.ajax({
                url: "/s/video/getVideoUploadStatus?id=" + encodeURIComponent(this.data.id),
                dataType: "json",
                success: function(data) {
                    self.status = data.status;
                    switch (data.status) {
                    case "REQUESTED_ID":
                    case "UPLOADED":
                        self.el.find(".webs-video-wrapper").addClass("video-uploading").removeClass("video-error");
                        break;
                    case "ERROR":
                        self.el.find(".webs-video-wrapper").addClass("video-error").removeClass("video-uploading");
                        break;
                    case "ENCODED":
                        setTimeout(function() {
                            var previewImg = self.el.find("img.webs-vidly-preview");
                            previewImg.attr("src", previewImg.attr("src").replace(/\?.*$/, "") + "?" + ( + (new Date()))); // Cache busting, fetch the 'actual' preview
                            self.el.find(".video-uploading").removeClass("video-uploading").removeClass("video-error");
                            previewImg.bind("load", function() {
                                setTimeout(function() {
                                    self.data.height = previewImg.height();
                                    self.data.ratio = previewImg.height() / previewImg.width();
                                    self.el.find(".webs-video-wrapper").height(self.data.height);
                                }, 1);
                            });
                        }, self.imageReloadDelay);
                        break;
                    }
                }
            });
        };

        module.manageStatusCheckThread = function() {
            var self = this;
            if (this.statusThread)
                clearInterval(this.statusThread);
            if (this.data.videoType != "browse")
                return;
            this.checkStatus();
            this.statusThread = setInterval(function() {
                self.checkStatus();
                if (self.status == "ENCODED" || self.status == "ERROR" || !self.el.parents("body").length) {
                    clearInterval(self.statusThread);
                }
            }, this.pollingInterval);
        };

        module.pollingInterval = 30 * 1000;
        module.imageReloadDelay = 30 * 1000;

        module.safeOneLoaded = function() {
            var self = this;
            this.bind("rendered", function() {
                self.manageStatusCheckThread();
            });
        };

    })(mod, ext);

    return ModuleClassLoader.register('video', mod, ext);
});




// title Module (custom)

define('module/title', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.title'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<h{{#if level}}{{level}}{{else}}3{{/if}} class="webs-title webs-title-align-{{align}} {{#if icon.enable}}webs-title-has_icon{{/if}}">\
        {{renderIcon icon}}\
        <div class="w-text {{#if font}}w-font-{{font}}{{/if}}"{{#style}} style="{{.}}"{{/style}} contenteditable="true" spellcheck="true" data-placeholder="{{#l}}webs.module.title.clicktoedit{{/l}}">{{#if html}}{{{html}}}{{else}}{{#l}}webs.module.title.clicktoedit{{/l}}{{/if}}</div>\
        </h{{#if level}}{{level}}{{else}}3{{/if}}>\
        ';

        // Module Default Data
        extend.defaultData = {
            "align": "left",
            "html": null,
            "level": 3,
            "icon": {
                "enable": false,
                "set": null,
                "slug": null,
                "position": null,
                "style": null
            }
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS


        // Edit JS
        /**
        * Initialize edit mode
        */
        module.edit = function(event) {
            this.enableIcon = true;

            this.controls = bldr.controls.create('title', {
                el: this.el.find('.webs-title'),
                data: this.data,
                enableIcon: this.enableIcon
            });
            this.controls.show();
            var editFrame = top.$("iframe#website")[0],
            editWindow = editFrame.contentWindow || editFrame.contentDocument.parentWindow;
            this.rangy = editWindow.rangy;
        };

        module.onDeactivate = function() {
            this.getData();
            if (this.controls) 
                this.controls.destroy();
        };

        module.getData = function() {
            var icon = this.data.icon;
            if (this.controls && this.controls.wizzy) {
                this.data = this.controls.getData();
            }
            if (!this.enableIcon) 
                this.data.icon = icon;
            return this.data;
        };

        module.beforeUndo = function() {
            if (this.controls.wizzy) {
                this.sessionRange = this.controls.wizzy.getRange();
            }
        };

        module.afterUndo = function() {
            var range = this.sessionRange;

            if (this.controls.wizzy) {
                this.controls.wizzy.setElement(this.el.find('.webs-title .w-text'));
            }

            if (!range || !range.isValid()) {
                range = this.rangy.createRange();
                range.selectNodeContents(this.el.find('.w-text')[0]);
                range.collapse();
            }

            this.rangy.getSelection().addRange(range);
        };

        module.safeOneLoaded = function() {
            var self = this;
            this.bind("rendered", function() {
                self.el.find("a").bind("click", function(e) {
                    e.preventDefault();
                });
            });
        };


    })(mod, ext);

    return ModuleClassLoader.register('title', mod, ext);
});




// google_map Module (custom)

define('module/google_map', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.google_map'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '{{#wrapInFrame}}\
        <div class="map_canvas {{#if frame}}webs-hasframe{{/if}} webs-align-{{align}}" \
        style="width:{{width}}px;max-width:100%;height:{{height}}px;">\
        \
        {{ address }}\
        \
        </div>\
        {{/wrapInFrame}}\
        ';

        // Module Default Data
        extend.defaultData = {
            "align": "center",
            "showControls": true,
            "getDirections": true,
            "showMarker": true,
            "useColor": true,
            "zoom": 12,
            "height": 200,
            "width": 200,
            "mapTypeId": "roadmap",
            "frame": false
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "plain": {
                "global": {
                    "js": "plain.view.js",
                    "css": "plain.view.less"
                },
                "default": true,
                "slug": "plain"
            }
        };
        extend.styles['plain']['global']['js'] = (function(module, extend) {


            /* plain.view.js */
            // see https://spreadsheets.google.com/pub?key=p9pdwsai2hDMsLkXsoM05KQ&gid=1 for mapping
            var languageMap = {
                "cs-CZ": "cs",
                "da-DK": "da",
                "de-AT": "de",
                "de-CH": "de",
                "de-DE": "de",
                "en-AU": "en-AU",
                "en-CA": "en",
                "en-GB": "en-GB",
                "en-IE": "en",
                "en-IN": "en",
                "en-NZ": "en",
                "en-SG": "en",
                "en-UK": "en",
                "en-US": "en",
                "es-ES": "es",
                "es-MX": "es",
                "es-US": "es",
                "fi-FI": "fi",
                "fr-BE": "fr",
                "fr-CA": "fr",
                "fr-CH": "fr",
                "fr-FR": "fr",
                "hi-IN": "hi",
                "it-CH": "it",
                "it-IT": "it",
                "ja-JP": "ja",
                "ko-KR": "ko",
                "nb-NO": "no",
                "nl-BE": "nl",
                "nl-NL": "nl",
                "no-NO": "no",
                "pl-PL": "pl",
                "pt-PT": "pt",
                "sv-SE": "sv",
                "tr-TR": "tr",
                "zh-CN": "zh-CN",
                "zh-TW": "zh-TW"
            };

            window._googleMapsApiKey = 'AIzaSyCYGaG3hI3yQdP7oH-o535wjeq4vxtpALU';

            // Google maps API requires a callback function
            window.onGoogleMapsLoaded = function() {
                module.googleMapsLoaded.resolve();
            };

            // Ensure google geo location and maps are loaded first
            module.googleCompetelyLoaded = $.Deferred();
            module.googleMapsLoaded = $.Deferred();

            extend.loadGoogleMaps = function() {

                // Geo location information
                var googleMapsGeoInfo = $.ajax({
                    type: 'GET',
                    url: 'http://www.google.com/jsapi',
                    dataType : "jsonp"
                });

                // Load google maps API
                var language = typeof(webs) != 'undefined' && languageMap[webs.locale] || "en",
                key = window._googleMapsApiKey,
                data = {
                    sensor: 'false',
                    callback: 'onGoogleMapsLoaded',
                    language: language 
                };
                if (key) 
                    data.key = key; // Only need a key in edit mode

                // Have to use google callback when complete
                $.ajax({
                    type: 'GET',
                    url: 'http://maps.googleapis.com/maps/api/js',
                    data: data,
                    dataType: 'script',
                    cache: true
                });

                // Ensure that we have both the maps API and location info before rendering our map
                $.when(googleMapsGeoInfo, module.googleMapsLoaded.promise()).then(function() {
                    module.googleCompetelyLoaded.resolve();
                });
            };

            module.oneLoaded = function() {
                this.displayMap();
            };

            module.displayMap = function() {

                var self = this;

                module.googleCompetelyLoaded.done(function() {

                    if (!self.data.center) {
                        // Center on the user's location if we could determine it
                        if (google.loader.ClientLocation) {
                            var clientLocation = new google.maps.LatLng(google.loader.ClientLocation.latitude, google.loader.ClientLocation.longitude);
                            self.data.center = self.data.marker = {
                                lat: clientLocation.lat(),
                                lng: clientLocation.lng() 
                            };
                        } else {
                            // default to showing DC
                            var dc = new google.maps.LatLng(38.895108, - 77.036369);
                            self.data.center = self.data.marker = {
                                lat: dc.lat(),
                                lng: dc.lng() 
                            };
                        }
                        self.displayMap();
                        return;
                    }

                    var frame = self.el.children();
                    var width = frame.width();
                    var el = self.el.find(".map_canvas")[0],
                    opts = self.mapOptions();

                    self.map = new google.maps.Map(el, opts);
                    self.displayMarker();
                    self.displayDirections();

                    // In IE8, when google maps are loaded, frame's width resets
                    // Forcing width on the map container after map is rendered
                    frame.width(width);
                });
            };

            module.setMapOptions = function() {
                this.map.setOptions(this.mapOptions());
                this.displayMarker();
                this.displayDirections();
            };

            module.displayMarker = function() {
                var opts = this.mapOptions(),
                addressHtml = '<div id="content">' + (this.data.address === undefined ? "Washington, DC" : this.data.address) + '</div>',
                bbl = new google.maps.InfoWindow({
                    content: addressHtml 
                }),
                self = this;

                if (this.marker) 
                    this.marker.setVisible(false);

                this.marker = new google.maps.Marker({
                    map: this.map,
                    title: this.data.address 
                });
                this.marker.setClickable(true);
                this.marker.setPosition(new google.maps.LatLng(this.data.marker.lat, this.data.marker.lng));
                this.marker.setVisible(!!this.data.showMarker);
                google.maps.event.addListener(this.marker, "click", function() {
                    bbl.open(self.map, self.marker);
                });
            };

            module.displayDirections = function() {
                var self = this;
                if (self.getDirections) 
                    self.getDirections.remove();

                self.getDirections = $("<div class='w-get-directions'>" + translate("webs.module.google_map.getdirections") + "</div>").
                appendTo(this.el.find(".map_canvas")).
                click(function() {
                    var location = self.data.address || (self.data.center.lat.toString() + " " + self.data.center.lng.toString()),
                    url = "http://maps.google.com/maps?q=to+" + encodeURIComponent(location);
                    window.open(url, "Directions");
                });

                if (!this.data.getDirections) 
                    self.getDirections.hide();
                else 
                    self.getDirections.show();

            };

            module.mapOptions = function() {
                return {
                    // user-changable options
                    center: new google.maps.LatLng(this.data.center.lat, this.data.center.lng),
                    zoom: this.data.zoom,

                    disableDefaultUI: !this.data.showControls,

                    styles: this.mapStyleOptions(),

                    // non-changable options
                    mapTypeId: this.data.mapTypeId,
                    mapTypeControl: true,

                    draggable: true,
                    scrollwheel: false,
                    disableDoubleClickZoom: false,
                    streetViewControl: false
                };
            };

            module.mapStyleOptions = function() {
                if (this.data.useColor) 
                    return undefined;
                return [
                {
                    "feature_type": "all",
                    "stylers": [ {
                        saturation: - 100 
                    }
                    ]
                }
                ];
            };

            // Finally, load google maps! (The setTimeout is here because IE was being weird)
            window.setTimeout(function() {
                extend.loadGoogleMaps();
            }, 1);
            return module;
        });
        extend.defaultStyle = extend.styles['plain'];

        // Register Toolbars
        bldr.toolbar.register('google_map', {
            "icon": "google_map",
            "items": [{
                "type": "textfield",
                "name": "address",
                "label": "Address",
                "placeholder": "placeholderAddress",
                "width": "200"
            }, {
                "icon": "border",
                "type": "framesets",
                "label": "Frame",
                "name": "frame",
                "width": 170
            }, {
                "type": "divider"
            }, {
                "type": "button-group",
                "sticky": true,
                "distinct": true,
                "items": [{
                    "label": "left",
                    "icon": "align_left",
                    "name": "align",
                    "value": "left"
                }, {
                    "label": "center",
                    "icon": "align_center",
                    "name": "align",
                    "value": "center"
                }, {
                    "label": "right",
                    "icon": "align_right",
                    "name": "align",
                    "value": "right"
                }
                ]
            }, {
                "type": "divider"
            }, {
                "type": "menu",
                "label": "Settings",
                "menuLabel": "Advanced Settings",
                "menuClassNames": "w-google_map-settings",
                "menu": [{
                    "label": "Show Controls",
                    "type": "switch-bare",
                    "display": "inline",
                    "name": "showControls"
                }, {
                    "label": "Get Directions",
                    "type": "switch-bare",
                    "display": "inline",
                    "name": "getDirections"
                }, {
                    "label": "Show Marker",
                    "type": "switch-bare",
                    "display": "inline",
                    "name": "showMarker"
                }, {
                    "label": "Color",
                    "type": "switch-bare",
                    "display": "inline",
                    "name": "useColor"
                }
                ]
            }
            ]
        });

        // View JS


        // Edit JS
        module.editMode = true;

        module.edit = function() {
            var self = this,
            $ = bldr.pageController.$;

            self.toolbar = bldr.toolbar.create("google_map");
            self.showToolbarFor(["showControls", "getDirections", "showMarker", "useColor", "address", "align"]);
            self.addFrameListener();
            self.showSizeControls();
            self.observeMapChanges();
            self.deactivateLinks();
            if (!this.data.address) {
                this.showHelptip(translate("webs.module.google_map.helptip"));
            }
        };

        // In editing mode only, deactivate Google links.
        module.deactivateLinks = function() {
            this.el.find(".map_canvas a").bind("click", function() {
                return false;
            });
        };

        // Custom render so we don't reload the map tiles 
        module.render = function() {
            if (this.el.find(".map_canvas").length === 0) {
                this.el.html(this.html());
                this.displayMap();
            } else if (!this.map) {
                this.displayMap();
            } else {
                this.setMapOptions();
            }
        };

        module.addFrameListener = function() {
            var self = this;

            self.toolbar.addListener("setFrame", function(slug) {
                if (typeof slug != "string") 
                    slug = null;
                self.data.frame = slug;
                self.frameChange();
            });
        };

        module.observeMapChanges = function() {
            var self = this;
            google.maps.event.addListener(this.map, 'zoom_changed', function() {
                self.data.zoom = this.getZoom();
            });
            google.maps.event.addListener(this.map, 'center_changed', function() {
                var center = this.getCenter();
                self.data.center = {
                    lat: center.lat(),
                    lng: center.lng() 
                };
            });
            google.maps.event.addListener(this.map, 'maptypeid_changed', function() {
                self.data.mapTypeId = this.getMapTypeId();
            });
        };

        module.fit = function(width, persist) {
            var map = this.el.find(".map_canvas"),
            frame = this.el.find(".webs-frame"),
            maxWidth = map.css("max-width");
            map.css({
                "max-width": "10000px"
            });
            var mapWidth = map.width(),
            newMapWidth = width - this.frameWidth();
            map.css({
                "max-width": maxWidth
            });

            this.maxWidth = width;

            if (!mapWidth || newMapWidth < mapWidth) {

                map.width(newMapWidth);
                frame.width(width);

                if (persist) {
                    this.data.width = newMapWidth;
                    this.dirty();
                }
            }
        };

        module.frameWidth = function() {
            var framePadding = 0,
            frame = this.el.find(".webs-frame"),
            map = this.el.find(".map_canvas");

            return frame.length > 0 ? frame.outerWidth() - map.parent().width() : 0;
        };

        module.sizeControlSides = function() {
            switch (this.data.align) {
            case "right":
                return ["left"];
            case "center":
                return ["right", "left"];
            default:
                return ["right"];
            }

        };

        module.removeSizeControls = function() {
            this.sizeControls && $.each(this.sizeControls, function(i, control) {
                control.remove();
            });
        };

        var $bodyTT = null,
        $sideTT = null;

        module.showSizeControls = function() {
            this.removeSizeControls();

            var self = this,
            mapE = self.el.find(".map_canvas"),
            $ = bldr.pageController.$,
            dragging = false,
            maxInnerWidth = this.maxWidth - this.frameWidth(),
            sides = this.sizeControlSides();

            if (!$bodyTT)
                $bodyTT = bldr.pageController.$.tooltip({
                    style: "warn",
                    content: translate("webs.module.google_map.maxwidth.container")
                });
            if (!$sideTT)
                $sideTT = bldr.pageController.$.tooltip({
                    style: "warn",
                    content: translate("webs.module.google_map.maxwidth.sidebar")
                });

            this.el.find(".map_canvas").css("overflow", "visible");
            this.sizeControls = $.map(sides, function(side) {
                var control = $("<div>").
                addClass("bldr-resize-handle bldr-resize-handle-" + side).
                appendTo(mapE).
                drag({
                    drag: false
                }).
                bind("dragStart", function(e, x, y) {
                    self.originalWidth = self.data.width;
                }).
                bind("drag", function(event, x, y) {
                    // need to rate-limit this callback
                    var controlLeft = 0, 
                    newWidth = 0;

                    self.data.height = x + 15;

                    if (side == "right") {
                        newWidth = (sides.length == 2 ? (y * 2 - self.originalWidth) : y) + 15;
                        controlLeft = self.data.width;
                    } else {
                        newWidth = self.originalWidth - (sides.length == 2 ? y * 2 : y);
                    }

                    self.data.width = Math.min(maxInnerWidth, newWidth);
                    if (self.data.width == maxInnerWidth) {
                        if (self.el.parents(".webs-sidebar").length) {
                            $sideTT.addClass("active");
                        } else {
                            $bodyTT.addClass("active");
                        }
                    } else {
                        $bodyTT.removeClass("active");
                        $sideTT.removeClass("active");
                    }
                    google.maps.event.trigger(self.map, 'resize');
                    mapE.height(self.data.height).width(self.data.width);
                    self.el.children().width(self.data.width + self.frameWidth());
                }).bind("dragEnd", function() {
                    $bodyTT.removeClass("active");
                    $sideTT.removeClass("active");
                    self.setMapOptions();
                });

                return control;
            });

            self.bind("blurred", function f() {
                $bodyTT.removeClass("active");
                $sideTT.removeClass("active");
                self.removeSizeControls();
                self.unbind("blurred", f);
            });
        };

        module.showToolbarFor = function(properties) {
            var listeners = {},
            data = {},
            self = this;
            $.each(properties, function(i, prop) {
                listeners[prop] = self.controlListenerFor(prop);
                data[prop] = self.data[prop];
            });
            self.toolbar.addListener(listeners);
            self.toolbar.show(data);
        };

        module.controlListenerFor = function(property) {
            var self = this;
            return function(val) {
                self.data[property] = val;
                self.changeFromControls(property);
            };
        };

        module.changeFromControls = function(changedProperty) {
            this.map.setOptions(this.mapOptions());
            this.displayMarker();

            var changeCallback = changedProperty + "Change";
            if (typeof this[changeCallback] == "function") 
                this[changeCallback]();
        };

        module.alignChange = function() {
            var alignedItem = this.el.find(".webs-frame, .map_canvas");
            //if(alignedItem.length === 0) alignedItem = this.el.find(".map_canvas");

            alignedItem.
            removeClass('webs-align-left webs-align-center webs-align-right').
            addClass("webs-align-" + this.data.align);

            this.showSizeControls();
        };

        module.getDirectionsChange = function() {
            this.displayDirections();
        };

        // re-render the whole module
        // after switching frames, new width (including frame) should equal old width
        module.frameChange = function() {
            var topLevel = this.el.find(".webs-frame"),
            frameWidth = this.frameWidth(),
            self = this;

            if (topLevel.length === 0) 
                topLevel = this.el.find(".map_canvas");

            this.el.empty();
            this.render();

            // get new top level
            var newFrameWidth = this.frameWidth(),
            newWidth = this.data.width - (newFrameWidth - frameWidth),
            ratio = newWidth / this.data.width,
            newHeight = Math.round(this.data.height * ratio, 10);
            this.data.height = newHeight;
            this.data.width = newWidth;

            this.el.empty();
            this.render();

            this.observeMapChanges();
            this.showSizeControls();
        };

        module.addressChange = function() {
            var self = this;
            this.geocodeAddress(function(loc) {
                self.data.center = self.data.marker = {
                    lat: loc.lat(),
                    lng: loc.lng() 
                };
                self.setMapOptions();
            }, function() {
                // what do we do on failure?
            });
        };

        module.geocodeAddress = function(success, failure) {
            var geocoder = new google.maps.Geocoder();
            geocoder.geocode({
                'address': this.data.address
            }, function(results, status) {
                if (status == google.maps.GeocoderStatus.OK) {
                    success(results[0].geometry.location);
                } else {
                    failure();
                }
            });
        };

        module.addMarker = function(map, location) {
            var marker = new google.maps.Marker({
                map: map,
                position: results[0].geometry.location
            });
        };

        module.getMinWidth = function() {
            return 250;
        };

        window._googleMapsApiKey = 'AIzaSyCYGaG3hI3yQdP7oH-o535wjeq4vxtpALU';


    })(mod, ext);

    return ModuleClassLoader.register('google_map', mod, ext);
});




// ads Module (widget)

define('module/ads', ['internal/sitebuilder/common/ModuleClassLoader'], function(ModuleClassLoader) {

    var mod = {}, ext = {};
    (function (module, extend) {

        // Show Cover?
        extend.shouldShowCover = function () {
            return false;
        };

        // Widget
        extend.isWidget = true;

        extend.popoverConfig = {
            "title": "Insert an Ad",
            "header": "Add a Google AdSense Ad to your webpage.",
            "button_text": "Save Changes",
            "preview_width": 200,
            "field_defs": [{
                "name": "id",
                "type": "text",
                "label": "AdSense ID",
                "hint": "Don't have Google AdSense? <a href='http://www.google.com/' target='_blank'>Sign Up</a>",
                "validation": {
                    "pattern": "^[A-Za-z0-9._%+-]+$",
                    "message": "Please enter your AdSense ID"
                }
            }, {
                "name": "type",
                "type": "menu",
                "label": "Type",
                "hint": "",
                "options": [{
                    "label": "Display Ads",
                    "value": "display"
                }, {
                    "label": "Text Ads",
                    "value": "text"
                }, {
                    "label": "Video Ads",
                    "value": "video"
                }, {
                    "label": "Link Units",
                    "value": "units"
                }
                ]
            }, {
                "name": "size",
                "type": "menu",
                "label": "Size",
                "hint": "",
                "options": [{
                    "label": "200 x 200px",
                    "value": "small"
                }, {
                    "label": "160 x 600px",
                    "value": "skyscraper"
                }, {
                    "label": "728 x 90px",
                    "value": "leaderboard"
                }, {
                    "label": "468 x 60px",
                    "value": "banner"
                }, {
                    "label": "250 x 250px",
                    "value": "square"
                }, {
                    "label": "236 x 280px",
                    "value": "large_rectangle"
                }, {
                    "label": "300 x 250px",
                    "value": "rectangle"
                }
                ]
            }
            ]
        };


        // Module HTML Template
        extend.template = '{{#case size "small"}}\
        <div class="ad small" name="submit" border="0"><span>Ad</span></div>\
        {{/case}}\
        \
        {{#case size "skyscraper"}}\
        <div class="ad skyscraper" name="submit" border="0"><span>Ad</span></div>\
        {{/case}}\
        \
        {{#case size "leaderboard"}}\
        <div class="ad leaderboard" name="submit" border="0"><span>Ad</span></div>\
        {{/case}}\
        \
        {{#case size "banner"}}\
        <div class="ad banner" name="submit" border="0"><span>Ad</span></div>\
        {{/case}}\
        \
        {{#case size "square"}}\
        <div class="ad square" name="submit" border="0"><span>Ad</span></div>\
        {{/case}}\
        \
        {{#case size "large_rectangle"}}\
        <div class="ad large_rectangle" name="submit" border="0"><span>Ad</span></div>\
        {{/case}}\
        \
        {{#case size "rectangle"}}\
        <div class="ad rectangle" name="submit" border="0"><span>Ad</span></div>\
        {{/case}}\
        ';

        // Module Default Data
        extend.defaultData = {
            "id": "",
            "type": "display",
            "size": "small"
        };

        // Module Styles
        extend.styles = {};
        extend.defaultStyle = extend.styles[''];

        module.getMinWidth = function() {
            return 250;
        };

        // Register Toolbars
        bldr.toolbar.register('ads', {
            "icon": "paypal",
            "items": [{
                "type": "button",
                "style": "blue",
                "label": "Edit Button",
                "name": "settingsDialog"
            }
            ]
        });

        // View JS


        // Edit JS


    })(mod, ext);

    return ModuleClassLoader.register('ads', mod, ext);

});


// email_list Module (custom)

define('module/email_list', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader'],
function($, ModuleClassLoader) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="w-email_list clearfix webs-composite-module">\
        {{submodule "title" title}}\
        \
        <form method="POST" action="http://link.members.freewebs.com/s/subscriptions/publicDispatcher">\
        <input type="hidden" name="ownerID" value="{{_site.id}}"/>\
        <input type="hidden" name="section" value="invite"/>\
        <input type="hidden" name="action" value="crash"/>\
        \
        <ul class="w-email_list-ul">\
        <li class="w-email_list-li input-container input-container-name"><input class="w-email_list-field" type="text" name="name" value="Name" data-defaultval="Name" /></li>\
        <li class="w-email_list-li input-container input-container-email"><input class="w-email_list-field" type="email" name="email" value="Email" data-defaultval="Email" /></li>\
        <li class="w-email_list-li input-container input-container-verify"><input class="w-email_list-field" type="email" name="email2" value="Verify Email" data-defaultval="Verify Email" /></li>\
        <li>{{submodule "text" text}}</li>\
        <li>{{submodule "button" button}}</li>\
        </ul>\
        </form>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "title": {
                "align": "left",
                "html": "Subscribe To My Website",
                "level": 3,
                "icon": {
                    "enable": false,
                    "set": null,
                    "slug": null,
                    "position": null,
                    "style": null
                }
            },
            "text": {
                "html": "<small>Subscribing allows you to get site updates. Your email address will be kept private.</small>"
            },
            "button": {
                "labelFor": "email_list-submit",
                "html": "Subscribe",
                "link": null,
                "linkInfo": {},
                "linkFrozen": true,
                "icon": {
                    "enable": false,
                    "set": null,
                    "slug": null,
                    "position": "left",
                    "style": null
                },
                "iconLeft": true,
                "iconStyle": "light",
                "bgcolor": null,
                "background": ""
            }
        };

        // SubModules
        extend.submodules = {
            "button": {
                "moduleType": "button"
            },
            "title": {
                "moduleType": "title"
            },
            "text": {
                "moduleType": "text"
            }
        };

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS
        module.oneLoaded = function() {
            var
            self = this,
            form = self.el.find("form"),
            $inputs = form.find(".input-container > input"),
            $inputName = form.find(".input-container-name > input"),
            $inputEmail = form.find(".input-container-email > input"),
            $inputVerify = form.find(".input-container-verify > input"),
            submitButton = self.el.find(".w-button"),
            errBlank = "Name can't be blank",
            errInvName = "Invalid Name",
            errInvEmail = "Invalid Email Address",
            errMatch = "Email does not match",
            errors = [errBlank, errInvName, errInvEmail, errMatch],
            emailRegexp = /^\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4})\s*$/;

            $inputs
            .focus(function() {
                var $el = $(this).removeClass("error"), val = $el.val();
                if (val === $el.data("defaultval") || $.inArray(val, errors) !== - 1) 
                    $el.val("");
            })
            .blur(function() {
                var $el = $(this);
                if ($el.val() === "") 
                    $el.val($el.data("defaultval"));
            });


            submitButton.click(function() {
                form.submit();
                return false;
            });
            form.submit(function() {

                var
                error = false,
                valName = $inputName.val(),
                valEmail = $inputEmail.val().toLowerCase(),
                valVerify = $inputVerify.val().toLowerCase();

                form.removeClass("error");
                $inputs.removeClass("error");

                if (valName.match(/^\s*$/)) {
                    error = true;
                    $inputName.val(errBlank).addClass("error");
                } else if (valName.match($inputName.data("defaultval"))) {
                    error = true;
                    $inputName.val(errInvName).addClass("error");
                }

                if (!valEmail.match(emailRegexp)) {
                    error = true;
                    $inputEmail.val(errInvEmail).addClass("error");
                }
                if (!valVerify.match(emailRegexp)) {
                    error = true;
                    $inputVerify.val(errInvEmail).addClass("error");
                }
                if (valVerify !== valEmail) {
                    error = true;
                    $inputVerify.val(errMatch).addClass("error");
                }

                if (!error) {
                    self.validEmail = valEmail;

                    // Force toLowerCase() on the vals since we're serializing the form and
                    // we have already validated so it wont mess with UX.
                    $inputEmail.val(valEmail);
                    $inputVerify.val(valEmail);
                    $.ajax({
                        "url": form.attr("action"),
                        "type": form.attr("method"),
                        "data": form.serialize(),
                        "dataType": "jsonp",
                        "contentType": 'application/json;charset=UTF-8',
                        "success": function(data) {
                            self.el.height(self.el.height());
                            if (data.error) {
                                self.showError(data.errorMsg);
                            } else {
                                self.showSuccess();
                            }
                        }
                    });
                }
                return false;
            });
        };

        // back-end error callback
        module.showError = function(errArray) {
            // We don't really need this since we already have front-end validation
        };

        module.showSuccess = function() {
            var formContainer = this.el.find(".w-email_list").after('<div class="w-email_list-success webs-text">' +
            '<h6><strong>Thank You</strong></h6>' +
            '<p>To complete the registration process, please click the link in the email we just sent to ' +
            '<span class="w-email_list-confirm">' + this.validEmail + '</span>.</p>' +
            '</div>');
            formContainer.remove();
        };


        // Edit JS
        module.onActivate = function(event) {
            require('internal/sitebuilder/builderChrome/toolbar').switchToolbar();
            this.el.find("form").find(".input-container > input").attr("disabled", "disabled");
        };

        module.attachCompositeControls = function() {
            var self = this;
            this.el.on("mousedown.submodules, mouseup.submodules", function(e) {
                // We only care about mouseup when we don't get the mousedown.
                // This happens on the 1st click, since the mousedown is before
                // the module is activated. Once we have any mouse event, we
                // can stop listening to mouseup, since we'll get the mousedown
                // next time.
                self.el.off("mouseup.submodules");

                var $submodule = $(e.target).closest(".webs-submodule");
                if ($submodule.length > 0) {
                    var slug = $submodule.attr("webs-submodule-slug")
                    self.activateSubmodule(slug);
                    if (e.type == "mouseup" && e.which == 3) {
                        // If we got a mouseup, it must be the first
                        // mouse event, which means that the mousedown
                        // happened before submodule activation, so we
                        // didn't get the helptip. Show it on the mouseup
                        // in this case, so that we always see it.
                        self.submoduleInstances[slug].rightClick();
                    }
                } else {
                    self.deactivateSubmodules();
                    require('internal/sitebuilder/builderChrome/toolbar').switchToolbar();
                }
            });
        };

        module.onDeactivate = function() {
            this.el.off("mousedown.submodules, mouseup.submodules");
        };

        module.getData = function() {
            if (this.submoduleInstances) {
                this.data.title = this.submoduleInstances.title.getData();
                this.data.text.html = this.submoduleInstances.text.getData().html;
                this.data.button = this.submoduleInstances.button.getData();
            }
            return this.data;
        };


    })(mod, ext);

    return ModuleClassLoader.register('email_list', mod, ext);
});




// html Module (custom)

define('module/html', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.html'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="w-custom-html">\
        {{#if html}}\
        <iframe class="preview_area" marginheight="0" marginwidth="0" frameborder="0" scrolling="no" allowtransparency=true></iframe>\
        <div class="cover"></div>\
        {{else}}\
        <div class="bldr-placeholder-element"><span class="placeholder-text">{{#l}}webs.module.html.editorPlaceholder{{/l}}</span></div>\
        {{/if}}\
        </div>';

        // Module Default Data
        extend.defaultData = {};

        // SubModules

        // Module Styles
        extend.styles = {
            "base": {
                "global": {
                    "css": "view.less"
                },
                "default": true,
                "slug": "base"
            }
        };
        if (!extend.styles['base']['global']) {
            extend.styles['base']['global'] = {};
        }
        extend.styles['base']['global']['js'] = null;
        extend.defaultStyle = extend.styles['base'];

        // Register Toolbars
        bldr.toolbar.register('html', {
            "icon": "html",
            "items": [{
                "type": "button",
                "style": "blue",
                "label": "Edit HTML",
                "name": "html",
                "dialog": {
                    "url": "/s/modules/html/v1.16/html.dialog.html",
                    "localizeHeading": "webs.module.html.dialog.heading",
                    "width": 650,
                    "height": 400
                }
            }
            ]
        });

        // View JS


        // Edit JS


        module.onActivate = function() {
            var self = this;
            this.bind("rendered", function() {
                self.attachControls();
            });
        };

        module.onDeactivate = function() {
            this.unbind("rendered");
        };

        module.showSettingsDialog = function() {
            if (this.shouldAllowCustomHTML()) {
                this.toolbar.click("html");
            } else {
                webs.showPremiumDialog('customhtml');
            }
        };

        // determine whether partner is the default partner (if webs.partner doesn't exist, assume default partner)
        module.isDefaultPartner = function() {
            return !webs.partner || webs.partner.isDefaultPartner;
        };

        // determine whether to allow user to edit module
        // non-webs users and premium users get to use custom html modules
        // grandfather free users with existing modules that contain html
        module.shouldAllowCustomHTML = function() {
            return !this.isDefaultPartner() || webs.site.premium || this.data.html;
        };

        module.attachControls = function() {
            var self = this,
            $ = bldr.pageController.$,
            data = self.data,
            container = self.el;

            self.toolbar = bldr.toolbar.create("html");

            if (self.shouldAllowCustomHTML()) {
                self.toolbar.show({
                    html: this.data.html || ""
                });
            }

            self.toolbar.addListener({
                html: function(data) {
                    self.data.html = data;
                    self.render();
                }
            });

            container.find(".w-custom-html").bind("dblclick", function(e) {
                self.showSettingsDialog();
                e.preventDefault();
                return false;
            });
        };


        // This method writes the html into an iframe document to prevent tampering with bldr edit
        module.renderInFrame = function(iframe) {
            var self = this;
            var idoc = self.getIframeDocument(iframe);

            setTimeout(function() {

                var fixedHTML = (self.data.html
                .replace(/wmode\s*=\s*('|")window("|')/i, "wmode='transparent'")
                .replace(/<param\s*name\s*=\s*('|")wmode("|')\s*value\s*=\s*('|")[a-zA-Z]("|')\s*\/?>/i, "<param name='wmode' value='opaque'/>")
                .replace(/<embed/, "<embed wmode='opaque'")
                .replace(/<\/object>/i, "<param name='wmode' value='opaque'></object>"));

                // _html_dat_ container will mirror this.el
                idoc.body.innerHTML = "<div id=\"_html_dat_\">" + fixedHTML + "</div>\n";
                idoc.body.innerHTML += "<style>html,body,div {margin:0;padding:0;}</style>\n";
                idoc.body.innerHTML += "<style>#_html_dat_ {" + self.getIframeStyle() + "}</style>";

                iframe.height(self.computeHeight(idoc));


                // <embed>s amd <object>s wind up on top of dialogs, the toaster, etc... Let's not have it do that:
                var embeds = idoc.getElementsByTagName("embed");
                for (var i = 0; i < embeds.length; ++i)
                    embeds[i].setAttribute("wmode", "opaque");
                var objects = idoc.getElementsByTagName("object");
                for (i = 0; i < objects.length; ++i) {
                    var wmodeParam = $(objects[i]).find("param[name='wmode'], param[name=WMode]");
                    if (wmodeParam.length)
                        wmodeParam.each(function(i, param) {
                            param.setAttribute("value", "opaque");
                        });
                    else
                        objects[i].innerHTML += "<param name='wmode' value='opaque'/>";
                }

                // Fix youtube's iframe overlay z-indexing
                var iframes = idoc.getElementsByTagName("iframe"), len = iframes.length, src, newsrc, _frame;
                var wmode = "wmode=transparent";
                for (i = 0; i < len; i++) {
                    _frame = iframes[i];
                    src = _frame.getAttribute("src");
                    if (src.indexOf("youtube") !== - 1 && src.indexOf("wmode") === - 1) {
                        if (src.indexOf('?') != - 1) {
                            var getQString = src.split('?');
                            var oldString = getQString[1];
                            var newString = getQString[0];
                            newsrc = newString + '?' + wmode + '&' + oldString;
                        } else 
                            newsrc = src + '?' + wmode;

                        _frame.setAttribute('src', newsrc);
                    }
                }
            }, 50);
        };

        module.render = function() {
            var self = this;
            var iframe = this.el.find(".preview_area");
            var idoc;

            if (!this.$html) 
                this.$html = $("html");

            if (this.renderedHTML !== undefined && this.renderedHTML === this.data.html) {
                if (iframe.length) {
                    idoc = this.getIframeDocument(iframe);
                    if (idoc.body.innerHTML.length !== 0) 
                        return;
                } else {
                    return;
                }
            }
            if ("html" in this.data && this.data.html !== "") {
                this.el.html(this.html());
                iframe = this.el.find(".preview_area");
                iframe.height(0);
                idoc = this.getIframeDocument(iframe);

                // We're starting with an empty doc, write its initial markup
                try {
                    idoc.open("text/html");
                    idoc.write("<!DOCTYPE html>\n<html><head></head><body></body></html>");
                    idoc.close();
                } catch (e) {}

                // Firefox (gecko) will call iframe load again, but not document.ready 
                if (this.$html.hasClass("gecko")) {
                    iframe.load(function() {
                        self.renderInFrame(iframe);
                    });
                } else {
                    $(idoc).ready(function() {
                        self.renderInFrame(iframe);
                    });
                }

            } else {
                this.el.html(this.html());
            }
            this.renderedHTML = this.data.html;
            this.trigger("rendered");
        };

        module.getIframeDocument = function(iframe) {
            var ifrm = iframe[0];
            var idoc = ((ifrm.contentWindow) ?
            ifrm.contentWindow :
            (ifrm.contentDocument.document) ?
            ifrm.contentDocument.document :
            ifrm.contentDocument
            ).document;
            return idoc;
        };

        module.computeHeight = function(doc) {
            var height = $(doc.body).height() || 40;
            return height;
        };

        module.getIframeStyle = function() {
            return (
            "background:transparent;" +
            "overflow:hidden;" +
            "word-wrap:break-word;" +
            "margin:0;" +
            "padding:0;" +
            "color:" + this.computeStyle("color") + ";" +
            "font-family:" + this.computeStyle("fontFamily") + ";" +
            "font-weight:" + this.computeStyle("fontWeight") + ";" +
            "font-size:" + this.computeStyle("fontSize") + ";"
            );
        };

        module.computeStyle = function(key) {
            if (this.el[0].currentStyle)
                return this.el[0].currentStyle[key];
            else if (document.defaultView && document.defaultView.getComputedStyle)
                return document.defaultView.getComputedStyle(this.el[0], "")[key];
            else
                return this.el[0].style[key];
        };

        module.fit = function(width, persist) {
            var self = this;
            if (this.oldWidth === width) 
                return;
            if ("renderedHTML" in this) {
                var iframe = this.el.find(".preview_area");
                if (!iframe.length) 
                    return;
                var idoc = this.getIframeDocument(iframe);
                iframe.width(width).height(0);
                iframe.height(this.computeHeight(idoc));
                this.oldWidth = width;
                this.dirty();
            }
        };

    })(mod, ext);

    return ModuleClassLoader.register('html', mod, ext);
});




// mobile_link Module (custom)

define('module/mobile_link', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader'],
function($, ModuleClassLoader) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<a class="webs-mobile webs-mobile-link-{{type}}" target="_top" href="{{url}}">{{name}}</a>\
        ';

        // Module Default Data
        extend.defaultData = {
            "name": "Webs",
            "url": "http://www.webs.com",
            "type": "external"
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS
        module.oneLoaded = function() {
            var linkURL = this.el.find('a').attr("href");

            // If neither http:// nor https:// are found, prepend http://
            if (linkURL.indexOf("http://") == - 1 && linkURL.indexOf("https://") == - 1) {
                linkURL = "http://" + linkURL;
                this.el.find('a').attr("href", linkURL);
            }
        };

        // Edit JS
        /**
        * Initialize edit mode
        */
        module.edit = function(event) {};

        module.initTooltip = function() {};

        module.onDeactivate = function() {};

        module.safeOneLoaded = function() {};


    })(mod, ext);

    return ModuleClassLoader.register('mobile_link', mod, ext);
});




// mobile_location Module (custom)

define('module/mobile_location', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader'],
function($, ModuleClassLoader) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<a class="webs-mobile webs-mobile-address href="http://maps.google.com/maps?q={{address.streetAddress}},+{{address.locality}},+{{address.region}}+{{address.postalCode}}">\
        <address class="adr">\
        <ul>\
        {{#with address}}\
        <li class="street-address">{{street1}}</li>\
        <li class="extended-address">{{street2}}</li>\
        <li><span class="locality">{{city}}</span>{{#if city}}, {{/if}}<span class="region">{{state}}</span> <span class="postal-code">{{postalCode}}</span></li>\
        {{/with}}\
        </ul>\
        </address>\
        </a>\
        ';

        // Module Default Data
        extend.defaultData = {
            "address": {
                "street1": "123 Main Street",
                "street2": "Suite 301",
                "city": "Gaithersburg",
                "state": "Maryland",
                "postalCode": "20878",
                "country": "United States"
            }
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS
        module.oneLoaded = function() {
            var self = this;
            var visitor = require([webs.props.staticServer + "/active-static/target/internal/sitebuilder/modules/common/visitor.js"],
            function(visitor) {

                // If this device is a mobile ios, use apple maps. Otherwise, google.
                var service = visitor.mobile && visitor.ios ? "apple" : "google";
                var linkBase = "http://maps." + service + ".com?q=";
                var $el = self.el;

                // Get the sections of the address from the module data.
                var addressParts = self.data.address;

                // Add each known part of the address to an array.
                var address = [];
                for (var part in addressParts) {
                    if (addressParts[part] !== "") {
                        address.push(addressParts[part]);
                    }
                }

                // Join the array by commma-space, and replace spaces with +.
                address = address.join(", ").replace(/\ /g, '+');

                // Set our new beautiful map href.
                $el.find("a").attr("href", linkBase + address);

            });
        };


        // Edit JS
        /**
        * Initialize edit mode
        */
        module.edit = function(event) {};

        module.initTooltip = function() {};

        module.onDeactivate = function() {};

        module.safeOneLoaded = function() {};


    })(mod, ext);

    return ModuleClassLoader.register('mobile_location', mod, ext);
});




// mobile_phone Module (custom)

define('module/mobile_phone', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader'],
function($, ModuleClassLoader) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return false;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return false;
        };



        // Module HTML Template
        extend.template = '<a class="webs-mobile webs-mobile-phone" href="tel:{{tel}}">{{tel}}</a>\
        ';

        // Module Default Data
        extend.defaultData = {
            "tel": "000-000-0000"
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS


        // Edit JS
        /**
        * Initialize edit mode
        */
        module.edit = function(event) {};

        module.initTooltip = function() {};

        module.onDeactivate = function() {};

        module.safeOneLoaded = function() {};


    })(mod, ext);

    return ModuleClassLoader.register('mobile_phone', mod, ext);
});




// mobile_social Module (custom)

define('module/mobile_social', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.mobile_social'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '{{#if facebook}}\
        <a class="webs-mobile webs-mobile-facebook" target="_top" href="http://m.facebook.com/{{username}}">\
        {{#l}}webs.module.mobile_social.facebook{{/l}}\
        </a>\
        {{/if}}\
        {{#if twitter}}\
        <a class="webs-mobile webs-mobile-twitter" target="_top" href="http://m.twitter.com/{{username}}">\
        {{#l}}webs.module.mobile_social.twitter{{/l}}\
        </a>\
        {{/if}}\
        {{#if pinterest}}\
        <a class="webs-mobile webs-mobile-pinterest" target="_top" href="http://m.pinterest.com/{{username}}">\
        {{#l}}webs.module.mobile_social.pinterest{{/l}}\
        </a>\
        {{/if}}\
        {{#if linkedin}}\
        <a class="webs-mobile webs-mobile-linkedin" target="_top" href="http://m.linkedin.com/in/{{username}}">\
        {{#l}}webs.module.mobile_social.linkedin{{/l}}\
        </a>\
        {{/if}} \
        ';

        // Module Default Data
        extend.defaultData = {
            "username": "webs",
            "facebook": true,
            "twitter": false,
            "pinterest": false,
            "linkedin": false
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS


        // Edit JS
        /**
        * Initialize edit mode
        */
        module.edit = function(event) {};

        module.initTooltip = function() {};

        module.onDeactivate = function() {};

        module.safeOneLoaded = function() {};

    })(mod, ext);

    return ModuleClassLoader.register('mobile_social', mod, ext);
});




// text_image Module (custom)

define('module/text_image', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader'],
function($, ModuleClassLoader) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="webs-text-image clearfix webs-composite-module">\
        {{submodule "image" image}}\
        {{submodule "text" text}}\
        </div>';

        // Module Default Data
        extend.defaultData = {
            "text": {
                "html": null
            },
            "image": {
                "url": null,
                "imageType": null,
                "align": "left",
                "width": 300,
                "height": 180,
                "imageWidth": 100,
                "frame": "default"
            }
        };

        // SubModules
        extend.submodules = {
            "image": {
                "moduleType": "image"
            },
            "text": {
                "moduleType": "text"
            }
        };

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS


        // Edit JS
        /**
        * Initialize edit mode
        */
        module.onActivate = function(event) {};

        module.attachCompositeControls = function() {
            var self = this;
            this.el.find(".webs-submodule-text").bind("click", function() {
                self.activateSubmodule("text");
            });
            this.el.find(".webs-submodule-image").bind("click", function() {
                self.activateSubmodule("image");
            });
        };

        module.onDeactivate = function() {};

        module.getData = function() {
            if (this.submoduleInstances) {
                this.data.text = this.submoduleInstances.text.getData();
                this.data.image = this.submoduleInstances.image.getData();
            }
            return this.data;
        };

        module.fit = function(width, persist) {
            var $container = this.el,
            wrapper = $container.find('.webs-image-wrapper-1'),
            data = this.data.image,
            diff;

            width = width - (wrapper.outerWidth(true) - wrapper.find('.webs-image-crop').width());
            diff = parseInt(data.width, 10) - width;

            if (diff > 0) {
                var newWidth = width,
                newHeight = parseInt(data.height / data.width * newWidth, 10);

                wrapper.width(width).height(newHeight);

                if (persist) {
                    data.width = wrapper.width();
                    data.height = newHeight;
                    wrapper.width(data.width).height(data.height);
                    this.dirty();
                }
            } else if (diff < 0) {
                wrapper.width(data.width).height(data.height);
                if (persist) {
                    this.dirty();
                }
            }
        };


    })(mod, ext);

    return ModuleClassLoader.register('text_image', mod, ext);
});




// widget_bank Module (custom)

define('module/widget_bank', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader'],
function($, ModuleClassLoader) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="w-widgetbank">\
        {{#if imgStr}}\
        {{{imgStr}}}\
        {{else}}\
        <div class="bldr-placeholder-element"></div>\
        {{/if}}\
        </div>';

        // Module Default Data
        extend.defaultData = {};

        // SubModules

        // Module Styles
        extend.styles = {
            "base": {
                "global": {
                    "css": "view.less"
                },
                "default": true,
                "slug": "base"
            }
        };
        if (!extend.styles['base']['global']) {
            extend.styles['base']['global'] = {};
        }
        extend.styles['base']['global']['js'] = null;
        extend.defaultStyle = extend.styles['base'];

        // Register Toolbars
        bldr.toolbar.register('dialog', {
            "icon": "widget_bank",
            "items": [{
                "type": "button",
                "style": "blue",
                "label": "Edit Widget",
                "name": "dialog",
                "dialog": {
                    "url": "/s/modules/widget_bank/v1.7/widget_bank.dialog.html",
                    "heading": "Edit Widget",
                    "width": 735,
                    "height": 474
                }
            }
            ]
        });

        // View JS


        // Edit JS
        $.extend(module, {
            onActivate: function() {
                var self = this;
                self.attachControls();
            },

            onDeactivate: function() {},

            showSettingsDialog: function() {
                this.toolbar.click("dialog");
            },

            attachControls: function() {
                var self = this;

                self.toolbar = bldr.toolbar.create("dialog");
                self.toolbar.show({
                    dialog: this.data 
                });

                self.toolbar.addListener({
                    dialog: function(data) {
                        self.data = data;
                        self.render();
                    }
                });

                self.el.find(".w-widgetbank").bind("dblclick", function(e) {
                    self.toolbar.click("dialog");
                    e.preventDefault();
                    return false;
                });
            },


            fit: function(width, persist) {}
        });

    })(mod, ext);

    return ModuleClassLoader.register('widget_bank', mod, ext);
});




// buckets Module (custom)

define('module/buckets', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.buckets'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="w-buckets-wrapper">\
        <div class="w-buckets cols{{numCols}} clearfix webs-composite-module">\
        {{#each cols}}\
        {{#show}}\
        <div class="w-bucket">\
        {{#ordered_list . ../../bucketContents}}\
        {{#if image}}{{submodule "image" image}}{{/if}}\
        {{#if title}}{{submodule "title" title}}{{/if}}\
        {{#if text}}{{submodule "text" text}}{{/if}}\
        {{#if button}}{{submodule "button" button}}{{/if}}\
        {{/ordered_list}}\
        </div>\
        {{/show}}\
        {{/each}}\
        </div>\
        </div>';

        // Module Default Data
        extend.defaultData = {
            "bucketContents": ["image", "title", "text", "button"],
            "numCols": 3,
            "cols": [{
                "image": {
                    "url": null,
                    "imageType": null,
                    "width": 300,
                    "height": 180,
                    "imageWidth": 100,
                    "border": "none",
                    "top": 0,
                    "inBucket": true,
                    "frame": "default"
                },
                "title": {
                    "align": "left",
                    "level": 3,
                    "icon": {
                        "enable": false,
                        "set": null,
                        "slug": null,
                        "position": null,
                        "style": null
                    }
                },
                "text": {},
                "button": {
                    "html": null,
                    "link": null,
                    "linkInfo": {
                        "linkType": "website",
                        "linkUrl": null,
                        "newWindow": false
                    },
                    "icon": {
                        "enable": false,
                        "set": null,
                        "slug": null,
                        "position": "left",
                        "style": null
                    },
                    "iconLeft": true,
                    "iconStyle": "light",
                    "bgcolor": null,
                    "background": ""
                },
                "show": true
            }, {
                "image": {
                    "url": null,
                    "imageType": null,
                    "width": 300,
                    "height": 180,
                    "imageWidth": 100,
                    "border": "none",
                    "top": 0,
                    "inBucket": true,
                    "frame": "default"
                },
                "title": {
                    "align": "left",
                    "level": 3,
                    "icon": {
                        "enable": false,
                        "set": null,
                        "slug": null,
                        "position": null,
                        "style": null
                    }
                },
                "text": {},
                "button": {
                    "html": null,
                    "link": null,
                    "linkInfo": {
                        "linkType": "website",
                        "linkUrl": null,
                        "newWindow": false
                    },
                    "icon": {
                        "enable": false,
                        "set": null,
                        "slug": null,
                        "position": "left",
                        "style": null
                    },
                    "iconLeft": true,
                    "iconStyle": "light",
                    "bgcolor": null,
                    "background": ""
                },
                "show": true
            }, {
                "image": {
                    "url": null,
                    "imageType": null,
                    "width": 300,
                    "height": 180,
                    "imageWidth": 100,
                    "border": "none",
                    "top": 0,
                    "inBucket": true,
                    "frame": "default"
                },
                "title": {
                    "align": "left",
                    "level": 3,
                    "icon": {
                        "enable": false,
                        "set": null,
                        "slug": null,
                        "position": null,
                        "style": null
                    }
                },
                "text": {},
                "button": {
                    "html": null,
                    "link": null,
                    "linkInfo": {
                        "linkType": "website",
                        "linkUrl": null,
                        "newWindow": false
                    },
                    "icon": {
                        "enable": false,
                        "set": null,
                        "slug": null,
                        "position": "left",
                        "style": null
                    },
                    "iconLeft": true,
                    "iconStyle": "light",
                    "bgcolor": null,
                    "background": ""
                },
                "show": true
            }, {
                "image": {
                    "url": null,
                    "imageType": null,
                    "width": 300,
                    "height": 180,
                    "imageWidth": 100,
                    "border": "none",
                    "top": 0,
                    "inBucket": true,
                    "frame": "default"
                },
                "title": {
                    "align": "left",
                    "level": 3,
                    "icon": {
                        "enable": false,
                        "set": null,
                        "slug": null,
                        "position": null,
                        "style": null
                    }
                },
                "text": {},
                "button": {
                    "html": null,
                    "link": null,
                    "linkInfo": {
                        "linkType": "website",
                        "linkUrl": null,
                        "newWindow": false
                    },
                    "icon": {
                        "enable": false,
                        "set": null,
                        "slug": null,
                        "position": "left",
                        "style": null
                    },
                    "iconLeft": true,
                    "iconStyle": "light",
                    "bgcolor": null,
                    "background": ""
                },
                "show": false
            }, {
                "image": {
                    "url": null,
                    "imageType": null,
                    "width": 300,
                    "height": 180,
                    "imageWidth": 100,
                    "border": "none",
                    "top": 0,
                    "inBucket": true,
                    "frame": "default"
                },
                "title": {
                    "align": "left",
                    "level": 3,
                    "icon": {
                        "enable": false,
                        "set": null,
                        "slug": null,
                        "position": null,
                        "style": null
                    }
                },
                "text": {},
                "button": {
                    "html": null,
                    "link": null,
                    "linkInfo": {
                        "linkType": "website",
                        "linkUrl": null,
                        "newWindow": false
                    },
                    "icon": {
                        "enable": false,
                        "set": null,
                        "slug": null,
                        "position": "left",
                        "style": null
                    },
                    "iconLeft": true,
                    "iconStyle": "light",
                    "bgcolor": null,
                    "background": ""
                },
                "show": false
            }
            ]
        };

        // SubModules
        extend.submodules = {
            "image": {
                "moduleType": "image"
            },
            "title": {
                "moduleType": "title"
            },
            "text": {
                "moduleType": "text"
            },
            "button": {
                "moduleType": "button"
            }
        };

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "instance": {
                    "css": "view.each.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS
        module.describeSubmodules = function() {
            var self = this,
            buckets = self.el.find(".w-bucket"),
            submoduleDescriptions = [];
            $.each(buckets, function(bucketIndex) {
                var bucket = buckets.eq(bucketIndex),
                submodules = bucket.find(".webs-submodule");
                $.each(submodules, function(submoduleIndex) {
                    var submodule = submodules.eq(submoduleIndex),
                    slug = submodule.attr("webs-submodule-slug");
                    submoduleDescriptions.push({
                        name: slug + bucketIndex,
                        el: submodule,
                        slug: slug,

                        data: self.data.cols[bucketIndex][slug]
                    });
                });
            });
            return submoduleDescriptions;
        };


        // Edit JS
        module.frameWidth = function() {
            if (!("image0" in this.submoduleInstances)) 
                return 0;
            var el = this.submoduleInstances["image0"].el,
            frame = el.find(".webs-frame"),
            image = el.find(".webs-image-wrapper-1");

            if (frame.length === 0) 
                return 0;
            return frame.outerWidth() - image.outerWidth();
        };

        module.onActivate = function() {};

        module.unbindOldControls = function() {
            if (!this.imageControls) 
                return;
            for (var i = 0, l = this.imageControls.length; i < l; i++)
                this.imageControls[i].hide();
        };

        module.showSettingsDialog = function() {
            var self = this;
            this.showDialog({
                "url": "/s/modules/buckets/v1.13/buckets.dialog.html",
                "heading": translate('webs.module.buckets.dialog.heading'),
                "width": 523,
                "height": 378,
                dialogData: {
                    "data": this.data
                },
                callback: function(data) {
                    self.data = data;
                    self.dirty();
                    var width = self.el.width();
                    self.render();
                    self.fit(width, true);
                }
            });
        };

        module.attachCompositeControls = function() {
            var self = this,
            buckets = self.el.find(".w-bucket"),
            imgWrappers = [],
            activeImageControls = null,
            maxCropHeight = 3000;

            this.unbindOldControls();
            self.imageControls = [];

            $.each(buckets, function(bucketIndex) {

                var bucket = buckets.eq(bucketIndex),
                submodules = bucket.find(".webs-submodule");

                imgWrappers[bucketIndex] = bucket.find(".webs-image-wrapper-1");
                self.data.cols[bucketIndex].image.inBucket = true;
                self.imageControls[bucketIndex] = bldr.controls.create('image', {
                    container: bucket,
                    el: bucket.find('.webs-image-wrapper-1'),
                    containerWidth: self.el.find('.w-bucket:eq(' + bucketIndex + ')').width() - self.frameWidth(),
                    data: self.data.cols[bucketIndex].image,
                    frameWidth: function() {
                        return self.frameWidth();
                    }
                });

                self.imageControls[bucketIndex].bind("setImage", function() {
                    // FIXME set image heights to match.
                });
                self.imageControls[bucketIndex].bind("setFrame", function(slug) {
                    var oldFrameWidth = self.frameWidth();

                    self.getData();
                    $.each(self.data.cols, function(i) {
                        self.data.cols[i].image.frame = slug;
                    });

                    self.render();
                    self.fit(self.totalWidth, true);
                    self.edit();
                    self.attachCompositeControls();
                });

                $.each(submodules, function(submoduleIndex) {

                    var submodule = submodules.eq(submoduleIndex),
                    slug = submodule.attr("webs-submodule-slug");

                    submodule.bind("click", function() {
                        if (slug == "image") {
                            if (activeImageControls) 
                                activeImageControls.hide();
                            self.deactivateSubmodules();
                            self.imageControls[bucketIndex].show();
                            activeImageControls = self.imageControls[bucketIndex];
                        } else {
                            if (activeImageControls) 
                                activeImageControls.hide();
                            activeImageControls = null;
                            self.activateSubmodule(slug + bucketIndex);
                        }
                    });
                });
            });

            this.el.find('.webs-image-wrapper-1').cropper({
                inBucket: true,
                target: this.el.find('.webs-image-crop img'),
                sides: ['bottom'],
                maxHeight: maxCropHeight,
                onMove: function(side, newWidth, newHeight) {
                    $.each(self.imageControls, function(col, imageControl) {
                        if (imageControl.zoomRange)
                            imageControl.zoomRange.attr('min', Math.max(imageControl.attributes.width, imageControl.bounds.ratio * newHeight));
                    });
                },
                onEnd: function(activeCropSide, finalWidth, finalHeight) {
                    $.each(self.data.cols, function(colnum, col) {
                        col.image.height = finalHeight;
                        if (colnum < self.data.numCols /*&& self.imageControls[colnum].$crop !== undefined*/
                        ) {
                            self.imageControls[colnum].height = finalHeight;
                            self.imageControls[colnum].readAttributes();
                        }
                    });
                    self.el.find('img').each(function(col, img) {
                        self.data.cols[col].image.top = parseInt($(img).css('top'), 10);
                    });
                }
            });
        };

        module.onDeactivate = function() {};

        module.getData = function() {
            var self = this;
            if (self.imageControls && self.submoduleInstances) {
                $.each(this.data.cols, function(colnum, col) {
                    if (col.show) {
                        if (typeof self.imageControls[colnum] !== 'undefined')
                            col.image = self.imageControls[colnum].getData() || col.image;
                        if (typeof self.submoduleInstances['title' + colnum] !== 'undefined')
                            col.title = self.submoduleInstances['title' + colnum].getData() || col.title;
                        if (typeof self.submoduleInstances['text' + colnum] !== 'undefined')
                            col.text = self.submoduleInstances['text' + colnum].getData() || col.text;
                        if (typeof self.submoduleInstances['button' + colnum] !== 'undefined')
                            col.button = self.submoduleInstances['button' + colnum].getData() || col.button;
                    }
                });
            }
            $.each(this.data.cols, function(i, col) {
                col.image.align = 'center';
            });
            return this.data;
        };

        module.fitLater = function(width, persist) {
            if (this._fitLaterTimeout) 
                return;

            var self = this;
            this._fitLaterTimeout = setTimeout(function() {
                self._fitLaterTimeout = false;
                self.fit(width, persist);
            }, 400);
        };

        module.fit = function(width, persist) {
            var $ = bldr.pageController.$,
            self = this,
            $container = this.el,
            buckets = $container.find('.w-bucket'),
            padding, border, eachWidth,
            changed = false;

            this.totalWidth = width;

            if (buckets.length == 1) {
                // If there's only one bucket, fit is a bit different
                eachWidth = width - (buckets.outerWidth() - buckets.width());
            } else {
                eachWidth = parseInt((width - 20 * (this.data.numCols - 1)) / this.data.numCols, 10);
            }

            if (isNaN(eachWidth)) {
                this.fitLater(width, persist);
                return;
            }

            buckets.each(function(col, bucket) {
                //Whole Bucket
                $(bucket).width(eachWidth);
                $(bucket).find(".webs-submodule").css({
                    "max-width": eachWidth + "px"
                });
                if ($.inArray('image', self.data.bucketContents) >= 0) {
                    //Image
                    var 
                    wrapper = $('.webs-image-wrapper-1', bucket),
                    data = self.data.cols[col].image,
                    newHeight,
                    thisWidth = eachWidth - self.frameWidth();
                    newHeight = parseInt(data.height / data.width * thisWidth, 10);
                    wrapper.width(thisWidth).height(newHeight);

                    if (persist) {
                        if (data.width != thisWidth) 
                            changed = true;
                        if (data.height != newHeight) 
                            changed = true;
                        data.width = thisWidth;
                        data.height = newHeight;
                    }
                }
            });
            if (persist && changed) 
                this.dirty();
        };

        module.getMinWidth = function() {
            var eachWidth = 60;
            return (this.data.numCols || 3) * eachWidth;
        };


    })(mod, ext);

    return ModuleClassLoader.register('buckets', mod, ext);
});




// google_checkout Module (widget)

define('module/google_checkout', ['internal/sitebuilder/common/ModuleClassLoader'], function(ModuleClassLoader) {

    var mod = {}, ext = {};
    (function (module, extend) {

        // Show Cover?
        extend.shouldShowCover = function () {
            return false;
        };

        // Widget
        extend.isWidget = true;

        extend.popoverConfig = {
            "title": "Edit Google Checkout Button",
            "header": "Create and add a Google Checkout button to your website.",
            "button_text": "Save Changes",
            "preview_width": 350,
            "field_defs": [{
                "name": "merchant_id",
                "validation": {
                    "pattern": "^((\\d{10})|(\\d{15}))$",
                    "message": "Merchant ID is invalid"
                },
                "type": "text",
                "label": "Google Merchant ID",
                "hint": "Don't have Google Checkout? <a href='https://checkout.google.com/' target='_blank'>Sign Up</a>"
            }, {
                "name": "item_name",
                "type": "text",
                "label": "Item Name",
                "hint": "(ex. Apple iPhone)"
            }, {
                "name": "description",
                "type": "text",
                "label": "Description",
                "hint": "(ex. This is a refurbished iPhone 4g.)"
            }, {
                "name": "price",
                "type": "number",
                "label": "Price",
                "hint": "(ex. 20.00) - Numbers Only.",
                "accepted_currency": [{
                    "label": "USD",
                    "value": "USD"
                }, {
                    "label": "EUR",
                    "value": "EUR"
                }, {
                    "label": "CAD",
                    "value": "CAD"
                }, {
                    "label": "GBP",
                    "value": "GBP"
                }
                ]
            }, {
                "name": "button_style",
                "type": "menu",
                "label": "Button Style",
                "hint": "",
                "options": [{
                    "label": "Small",
                    "value": "small"
                }, {
                    "label": "Medium",
                    "value": "medium"
                }, {
                    "label": "Large",
                    "value": "large"
                }
                ]
            }, {
                "name": "color_scheme",
                "type": "menu",
                "label": "Color Scheme",
                "hint": "",
                "options": [{
                    "label": "Transparent",
                    "value": "trans"
                }, {
                    "label": "White",
                    "value": "white"
                }
                ]
            }
            ]
        };


        // Module HTML Template
        extend.template = '<div class="google_checkout {{#unless merchant_id}}disabled{{/unless}}">\
        {{#case button_style "small"}}\
        <img alt="Buy Now" src="https://checkout.google.com/buttons/checkout.gif?merchant_id={{merchant_id}}&w=160&h=43&style={{color_scheme}}&variant=text&loc=en_US"/>\
        {{/case}}\
        \
        {{#case button_style "medium"}}\
        <img alt="Buy Now" src="https://checkout.google.com/buttons/checkout.gif?merchant_id={{merchant_id}}&w=168&h=44&style={{color_scheme}}&variant=text&loc=en_US"/>\
        {{/case}}\
        \
        {{#case button_style "large"}}\
        <img alt="Buy Now" src="https://checkout.google.com/buttons/checkout.gif?merchant_id={{merchant_id}}&w=180&h=46&style={{color_scheme}}&variant=text&loc=en_US"/>\
        {{/case}}\
        </div>';

        // Module Default Data
        extend.defaultData = {
            "merchant_id": "",
            "item_name": "",
            "description": "",
            "price": 0,
            "price_currency": "USD",
            "button_style": "medium",
            "color_scheme": "trans"
        };

        // Module Styles
        extend.styles = {
            "base": {
                "global": {
                    "css": "view.less"
                },
                "slug": "base"
            }
        };
        if (!extend.styles['base']['global']) {
            extend.styles['base']['global'] = {};
        }
        extend.styles['base']['global']['js'] = null;
        extend.defaultStyle = extend.styles['base'];

        module.getMinWidth = function() {
            return 250;
        };

        // Register Toolbars
        bldr.toolbar.register('google_checkout', {
            "icon": "google_checkout",
            "items": [{
                "type": "button",
                "style": "blue",
                "label": "Edit Button",
                "name": "settingsDialog"
            }
            ]
        });

        // View JS


        // Edit JS


    })(mod, ext);

    return ModuleClassLoader.register('google_checkout', mod, ext);

});


// donate Module (widget)

// breaks if require is used
define('module/donate', ['internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.donate'], function(ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function (module, extend) {

        // Show Cover?
        extend.shouldShowCover = function () {
            return false;
        };

        // Widget
        extend.isWidget = true;

        extend.popoverConfig = {
            "title": "webs.module.donate.title",
            "header": "webs.module.donate.header",
            "button_text": "webs.module.donate.buttontext",
            "preview_width": 350,
            "field_defs": [{
                "name": "paypal_username",
                "type": "text",
                "label": "webs.module.donate.field.username.label",
                "hint": "webs.module.donate.field.username.hint",
                "validation": {
                    "pattern": "^\\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4})\\s*$",
                    "message": "webs.module.donate.field.username.validationmsg"
                }
            }, {
                "name": "donation_for",
                "type": "text",
                "label": "webs.module.donate.field.donation.label",
                "hint": "webs.module.donate.field.donation.hint",
                "validation": {
                    "pattern": "[^\\s]",
                    "message": "webs.module.donate.field.donation.validationmsg"
                }
            }, {
                "name": "donation_amount",
                "type": "currency",
                "allow_blank": true,
                "label": "webs.module.donate.field.donationamt.label",
                "hint": "webs.module.donate.field.donationamt.hint",
                "accepted_currency": [{
                    "label": "USD",
                    "value": "USD"
                }, {
                    "label": "EUR",
                    "value": "EUR"
                }, {
                    "label": "CAD",
                    "value": "CAD"
                }, {
                    "label": "GBP",
                    "value": "GBP"
                }, {
                    "label": "AUD",
                    "value": "AUD"
                }, {
                    "label": "DKK",
                    "value": "DKK"
                }, {
                    "label": "SEK",
                    "value": "SEK"
                }
                ]
            }, {
                "name": "button_style",
                "type": "menu",
                "label": "webs.module.donate.field.buttonstyle.label",
                "hint": "",
                "options": [{
                    "label": "webs.module.donate.field.buttonstyle.opt.small",
                    "value": "small"
                }, {
                    "label": "webs.module.donate.field.buttonstyle.opt.large",
                    "value": "large"
                }, {
                    "label": "webs.module.donate.field.buttonstyle.opt.largecc",
                    "value": "large_cc"
                }
                ]
            }, {
                "name": "spacer",
                "type": "spacer"
            }
            ]
        };
        extend.popoverConfig['title'] = translate('webs.module.donate.title');
        extend.popoverConfig['header'] = translate('webs.module.donate.header');
        extend.popoverConfig['button_text'] = translate('webs.module.donate.buttontext');
        extend.popoverConfig['field_defs'][0]['label'] = translate('webs.module.donate.field.username.label');
        extend.popoverConfig['field_defs'][0]['hint'] = translate('webs.module.donate.field.username.hint');
        extend.popoverConfig['field_defs'][0]['validation']['message'] = translate('webs.module.donate.field.username.validationmsg');
        extend.popoverConfig['field_defs'][1]['label'] = translate('webs.module.donate.field.donation.label');
        extend.popoverConfig['field_defs'][1]['hint'] = translate('webs.module.donate.field.donation.hint');
        extend.popoverConfig['field_defs'][1]['validation']['message'] = translate('webs.module.donate.field.donation.validationmsg');
        extend.popoverConfig['field_defs'][2]['label'] = translate('webs.module.donate.field.donationamt.label');
        extend.popoverConfig['field_defs'][2]['hint'] = translate('webs.module.donate.field.donationamt.hint');
        extend.popoverConfig['field_defs'][3]['label'] = translate('webs.module.donate.field.buttonstyle.label');
        extend.popoverConfig['field_defs'][3]['options'][0]['label'] = translate('webs.module.donate.field.buttonstyle.opt.small');
        extend.popoverConfig['field_defs'][3]['options'][1]['label'] = translate('webs.module.donate.field.buttonstyle.opt.large');
        extend.popoverConfig['field_defs'][3]['options'][2]['label'] = translate('webs.module.donate.field.buttonstyle.opt.largecc');


        // Module HTML Template
        extend.template = '	{{#case button_style "small"}}\
        <img name="submit" border="0" src="https://www.paypal.com/{{locale}}/i/btn/btn_donate_SM.gif" alt="{{#l}}webs.module.donate.slogan{{/l}}"/>\
        {{/case}}\
        \
        {{#case button_style "large"}}\
        <img name="submit" border="0" src="https://www.paypal.com/{{locale}}/i/btn/btn_donate_LG.gif" alt="{{#l}}webs.module.donate.slogan{{/l}}"/>\
        {{/case}}\
        \
        {{#case button_style "large_cc"}}\
        <img name="submit" border="0" src="https://www.paypal.com/{{locale}}/i/btn/btn_donateCC_LG.gif" alt="{{#l}}webs.module.donate.slogan{{/l}}"/>\
        {{/case}}';

        // Module Default Data
        extend.defaultData = {
            "paypal_username": "",
            "donation_for": "",
            "donation_amount": "",
            "donation_amount_currency": "",
            "button_style": "large_cc"
        };

        // Module Styles
        extend.styles = {};
        extend.defaultStyle = extend.styles[''];

        module.getMinWidth = function() {
            return 250;
        };

        // Register Toolbars
        bldr.toolbar.register('donate', {
            "icon": "donate",
            "items": [{
                "type": "button",
                "style": "blue",
                "label": "Edit Button",
                "name": "settingsDialog"
            }
            ]
        });

        // View JS
        if (!extend.defaultData) {
            extend.defaultData = {
                "locale": "en_US"
            };
        }

        if (typeof(webs) !== 'undefined' && webs.locale) {
            var languageMap = {
                "da-DK": "da_DK",
                "de-DE": "de_DE",
                "en-AU": "en_AU",
                "en-GB": "en_GB",
                "en-US": "en_US",
                "es-ES": "es_ES",
                "fr-CA": "fr_CA",
                "fr-FR": "fr_FR",
                "it-IT": "it_IT",
                "ja-JP": "ja_JP",
                "nl-NL": "nl_NL",
                "no-NO": "no_NO",
                "pl-PL": "pl_PL",
                "sv-SE": "sv_SE",
                "tr-TR": "tr_TR",
                "zh-CN": "zh_CN",
                "zh-TW": "zh_TW",
                "zz-ZZ": "nl_NL"
            };

            extend.defaultData.locale = languageMap[webs.locale] || "en_US";
        }

        // Edit JS
        module.safeOneLoaded = function() {
            this.translateDefaults();
            var self = this;
            setTimeout(function() {
                self.dirty();
            }, 0);
        };

        module.translateDefaults = function() {
            if (this.data.donation_amount_currency == "") {
                this.data.donation_amount_currency = translate("webs.module.donate.default_currency");
            }
        };

    })(mod, ext);

    return ModuleClassLoader.register('donate', mod, ext);

});


// fb-comments Module (widget)

define('module/fb-comments', ['internal/sitebuilder/common/ModuleClassLoader'], function(ModuleClassLoader) {

    var mod = {}, ext = {};
    (function (module, extend) {

        // Show Cover?
        extend.shouldShowCover = function () {
            return false;
        };

        // Widget
        extend.isWidget = true;


        module.helptip = {
            "fieldName": "url",
            "message": "Enter your Facebook URL above"
        };

        // Module HTML Template
        extend.template = '<div class=\'fb-comments-wrap\' id="fb-comments-{{id}}">\
        <div class="fb-comments" data-href="{{url}}" data-num-posts="{{numPosts}}" data-width="{{_width}}" data-colorscheme="{{colorScheme}}"></div>\
        <script>\
        (function(d, s, id) {\
        if($("#fb-root").length == 0){\
        $("<div id=\'fb-root\'></div>").appendTo("body");\
        }\
        \
        if(window.FB){\
        var el = document.getElementById("fb-comments-{{id}}");\
        FB.XFBML.parse(el);\
        } else {\
        var js, fjs = d.getElementsByTagName(s)[0];\
        js = d.createElement(s); js.id = id;\
        var locale = ((typeof webs !== "undefined" && webs.locale) || "en-US").replace("-","_");\
        if (locale === "zz_ZZ") {\
        locale = "fr_FR";\
        }\
        js.src = "//connect.facebook.net/" + locale + "/all.js#xfbml=1";\
        fjs.parentNode.insertBefore(js, fjs);\
        }\
        }(document, \'script\', \'facebook-jssdk\'));\
        </script>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "numPosts": "2",
            "colorScheme": "light"
        };

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        module.getMinWidth = function() {
            return 250;
        };

        // Register Toolbars
        bldr.toolbar.register('fb-comments', {
            "icon": "fb-comments",
            "items": [{
                "type": "menu",
                "menuPosition": "left",
                "label": "Settings",
                "width": 178,
                "menuClassNames": "w-fb-comments-advanced",
                "menu": [{
                    "type": "dropdown",
                    "label": "# of Posts",
                    "name": "numPosts",
                    "display": "inline",
                    "width": 60,
                    "options": [{
                        "label": "1",
                        "value": "1"
                    }, {
                        "label": "2",
                        "value": "2"
                    }, {
                        "label": "3",
                        "value": "3"
                    }, {
                        "label": "4",
                        "value": "4"
                    }, {
                        "label": "5",
                        "value": "5"
                    }, {
                        "label": "6",
                        "value": "6"
                    }, {
                        "label": "7",
                        "value": "7"
                    }, {
                        "label": "8",
                        "value": "8"
                    }, {
                        "label": "9",
                        "value": "9"
                    }, {
                        "label": "10",
                        "value": "10"
                    }
                    ]
                }, {
                    "type": "dropdown",
                    "label": "Color Scheme",
                    "name": "colorScheme",
                    "display": "inline",
                    "width": 60,
                    "options": [{
                        "label": "Light",
                        "value": "light"
                    }, {
                        "label": "Dark",
                        "value": "dark"
                    }
                    ]
                }
                ]
            }
            ]
        });

        // View JS


        // Edit JS
        module.render = function() {
            if (!this.data.url) {
                this.data.url = webs.page.url;
                this.data.id = Math.random().toString().slice(2);
                this.dirty();
            }
            this.parent.parent.fn.render.call(this);
        };

        module.safeOneLoaded = function() {
            this.chooseDarkOrLightStyle('dark', 'light');
        };


    })(mod, ext);

    return ModuleClassLoader.register('fb-comments', mod, ext);

});


// horizontal_rule Module (custom)

define('module/horizontal_rule', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader'],
function($, ModuleClassLoader) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="webs-hr webs-align-{{position}}" style="width: {{width}}; margin-top: {{margin}}px; margin-bottom: {{margin}}px;">\
        <div class="webs-hr-rule">\
        <div class="webs-hr-left-container webs-hr-container">\
        <div class="webs-hr-left webs-hr-bg"></div>\
        </div>\
        <div class="webs-hr-left-spacer-container webs-hr-container">\
        <div class="webs-hr-left-spacer webs-hr-bg"></div>\
        </div>\
        <div class="webs-hr-center-container webs-hr-container">\
        <div class="webs-hr-center webs-hr-bg"></div>\
        </div>\
        <div class="webs-hr-right-spacer-container webs-hr-container">\
        <div class="webs-hr-right-spacer webs-hr-bg"></div>\
        </div>\
        <div class="webs-hr-right-container webs-hr-container">\
        <div class="webs-hr-right webs-hr-bg"></div>\
        </div>\
        </div>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "width": "100%",
            "margin": 1,
            "position": "center"
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "base": {
                "global": {
                    "css": "view.less"
                },
                "slug": "base"
            },
            "dark": {
                "global": {
                    "css": "view.dark.less"
                },
                "inherit": "base",
                "slug": "dark"
            },
            "light": {
                "global": {
                    "css": "view.light.less"
                },
                "inherit": "base",
                "slug": "light"
            },
            "dark-1": {
                "inherit": "dark",
                "slug": "dark-1"
            },
            "dark-2": {
                "inherit": "dark",
                "slug": "dark-2"
            },
            "dark-3": {
                "inherit": "dark",
                "slug": "dark-3"
            },
            "dark-4": {
                "inherit": "dark",
                "slug": "dark-4"
            },
            "dark-5": {
                "inherit": "dark",
                "slug": "dark-5"
            },
            "dark-6": {
                "inherit": "dark",
                "slug": "dark-6"
            },
            "dark-7": {
                "inherit": "dark",
                "slug": "dark-7"
            },
            "dark-8": {
                "inherit": "dark",
                "slug": "dark-8"
            },
            "dark-9": {
                "inherit": "dark",
                "slug": "dark-9"
            },
            "dark-10": {
                "inherit": "dark",
                "slug": "dark-10"
            },
            "dark-11": {
                "inherit": "dark",
                "slug": "dark-11"
            },
            "dark-12": {
                "inherit": "dark",
                "slug": "dark-12"
            },
            "dark-13": {
                "inherit": "dark",
                "slug": "dark-13"
            },
            "light-1": {
                "inherit": "light",
                "default": true,
                "slug": "light-1"
            },
            "light-2": {
                "inherit": "light",
                "slug": "light-2"
            },
            "light-3": {
                "inherit": "light",
                "slug": "light-3"
            },
            "light-4": {
                "inherit": "light",
                "slug": "light-4"
            },
            "light-5": {
                "inherit": "light",
                "slug": "light-5"
            },
            "light-6": {
                "inherit": "light",
                "slug": "light-6"
            },
            "light-7": {
                "inherit": "light",
                "slug": "light-7"
            },
            "light-8": {
                "inherit": "light",
                "slug": "light-8"
            },
            "light-9": {
                "inherit": "light",
                "slug": "light-9"
            },
            "light-10": {
                "inherit": "light",
                "slug": "light-10"
            },
            "light-11": {
                "inherit": "light",
                "slug": "light-11"
            },
            "light-12": {
                "inherit": "light",
                "slug": "light-12"
            },
            "light-13": {
                "inherit": "light",
                "slug": "light-13"
            }
        };
        if (!extend.styles['base']['global']) {
            extend.styles['base']['global'] = {};
        }
        extend.styles['base']['global']['js'] = null;
        if (!extend.styles['dark']['global']) {
            extend.styles['dark']['global'] = {};
        }
        extend.styles['dark']['global']['js'] = null;
        if (!extend.styles['light']['global']) {
            extend.styles['light']['global'] = {};
        }
        extend.styles['light']['global']['js'] = null;
        if (!extend.styles['dark-1']['global']) {
            extend.styles['dark-1']['global'] = {};
        }
        extend.styles['dark-1']['global']['js'] = null;
        if (!extend.styles['dark-2']['global']) {
            extend.styles['dark-2']['global'] = {};
        }
        extend.styles['dark-2']['global']['js'] = null;
        if (!extend.styles['dark-3']['global']) {
            extend.styles['dark-3']['global'] = {};
        }
        extend.styles['dark-3']['global']['js'] = null;
        if (!extend.styles['dark-4']['global']) {
            extend.styles['dark-4']['global'] = {};
        }
        extend.styles['dark-4']['global']['js'] = null;
        if (!extend.styles['dark-5']['global']) {
            extend.styles['dark-5']['global'] = {};
        }
        extend.styles['dark-5']['global']['js'] = null;
        if (!extend.styles['dark-6']['global']) {
            extend.styles['dark-6']['global'] = {};
        }
        extend.styles['dark-6']['global']['js'] = null;
        if (!extend.styles['dark-7']['global']) {
            extend.styles['dark-7']['global'] = {};
        }
        extend.styles['dark-7']['global']['js'] = null;
        if (!extend.styles['dark-8']['global']) {
            extend.styles['dark-8']['global'] = {};
        }
        extend.styles['dark-8']['global']['js'] = null;
        if (!extend.styles['dark-9']['global']) {
            extend.styles['dark-9']['global'] = {};
        }
        extend.styles['dark-9']['global']['js'] = null;
        if (!extend.styles['dark-10']['global']) {
            extend.styles['dark-10']['global'] = {};
        }
        extend.styles['dark-10']['global']['js'] = null;
        if (!extend.styles['dark-11']['global']) {
            extend.styles['dark-11']['global'] = {};
        }
        extend.styles['dark-11']['global']['js'] = null;
        if (!extend.styles['dark-12']['global']) {
            extend.styles['dark-12']['global'] = {};
        }
        extend.styles['dark-12']['global']['js'] = null;
        if (!extend.styles['dark-13']['global']) {
            extend.styles['dark-13']['global'] = {};
        }
        extend.styles['dark-13']['global']['js'] = null;
        if (!extend.styles['light-1']['global']) {
            extend.styles['light-1']['global'] = {};
        }
        extend.styles['light-1']['global']['js'] = null;
        if (!extend.styles['light-2']['global']) {
            extend.styles['light-2']['global'] = {};
        }
        extend.styles['light-2']['global']['js'] = null;
        if (!extend.styles['light-3']['global']) {
            extend.styles['light-3']['global'] = {};
        }
        extend.styles['light-3']['global']['js'] = null;
        if (!extend.styles['light-4']['global']) {
            extend.styles['light-4']['global'] = {};
        }
        extend.styles['light-4']['global']['js'] = null;
        if (!extend.styles['light-5']['global']) {
            extend.styles['light-5']['global'] = {};
        }
        extend.styles['light-5']['global']['js'] = null;
        if (!extend.styles['light-6']['global']) {
            extend.styles['light-6']['global'] = {};
        }
        extend.styles['light-6']['global']['js'] = null;
        if (!extend.styles['light-7']['global']) {
            extend.styles['light-7']['global'] = {};
        }
        extend.styles['light-7']['global']['js'] = null;
        if (!extend.styles['light-8']['global']) {
            extend.styles['light-8']['global'] = {};
        }
        extend.styles['light-8']['global']['js'] = null;
        if (!extend.styles['light-9']['global']) {
            extend.styles['light-9']['global'] = {};
        }
        extend.styles['light-9']['global']['js'] = null;
        if (!extend.styles['light-10']['global']) {
            extend.styles['light-10']['global'] = {};
        }
        extend.styles['light-10']['global']['js'] = null;
        if (!extend.styles['light-11']['global']) {
            extend.styles['light-11']['global'] = {};
        }
        extend.styles['light-11']['global']['js'] = null;
        if (!extend.styles['light-12']['global']) {
            extend.styles['light-12']['global'] = {};
        }
        extend.styles['light-12']['global']['js'] = null;
        if (!extend.styles['light-13']['global']) {
            extend.styles['light-13']['global'] = {};
        }
        extend.styles['light-13']['global']['js'] = null;
        extend.defaultStyle = extend.styles['light-1'];

        // Register Toolbars
        bldr.toolbar.register('horizontal_rule', {
            "icon": "horizontal_rule",
            "items": [{
                "type": "radio-set",
                "title": "",
                "label": "Styles",
                "menuLabel": "Divider Styles",
                "name": "style",
                "menuPosition": "center",
                "menuClassNames": "w-radio-set-panel",
                "menuItemClassNames": "w-radio-set-item",
                "menuOptionSelectClass": "w-radio-set-select",
                "menuOptionClass": "w-radio-set",
                "dropdownTitle": "Divider Color",
                "selectedValue": "Light",
                "options": {
                    "Light": [{
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_dk_11.png",
                        "value": "dark-11"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_dk_12.png",
                        "value": "dark-12"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_dk_13.png",
                        "value": "dark-13"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_dk_01.png",
                        "value": "dark-1"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_dk_02.png",
                        "value": "dark-2"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_dk_03.png",
                        "value": "dark-3"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_dk_04.png",
                        "value": "dark-4"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_dk_05.png",
                        "value": "dark-5"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_dk_06.png",
                        "value": "dark-6"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_dk_07.png",
                        "value": "dark-7"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_dk_08.png",
                        "value": "dark-8"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_dk_09.png",
                        "value": "dark-9"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_dk_10.png",
                        "value": "dark-10"
                    }
                    ],
                    "Dark": [{
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_lt_11.png",
                        "value": "light-11"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_lt_12.png",
                        "value": "light-12"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_lt_13.png",
                        "value": "light-13"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_lt_01.png",
                        "value": "light-1"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_lt_02.png",
                        "value": "light-2"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_lt_03.png",
                        "value": "light-3"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_lt_04.png",
                        "value": "light-4"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_lt_05.png",
                        "value": "light-5"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_lt_06.png",
                        "value": "light-6"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_lt_07.png",
                        "value": "light-7"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_lt_08.png",
                        "value": "light-8"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_lt_09.png",
                        "value": "light-9"
                    }, {
                        "icon": "/s/modules/horizontal_rule/v1.14/icons/div_lt_10.png",
                        "value": "light-10"
                    }
                    ]
                }
            }, {
                "type": "menu",
                "label": "Advanced",
                "width": 225,
                "menuClassNames": "w-hrule-advanced",
                "menu": [{
                    "type": "dropdown",
                    "label": "Margin",
                    "name": "margin",
                    "width": 60,
                    "display": "inline",
                    "options": [{
                        "label": 1,
                        "value": 1
                    }, {
                        "label": 2,
                        "value": 3
                    }, {
                        "label": 3,
                        "value": 5
                    }, {
                        "label": 4,
                        "value": 8
                    }, {
                        "label": 5,
                        "value": 12
                    }
                    ]
                }, {
                    "type": "dropdown",
                    "label": "Width",
                    "name": "width",
                    "width": 80,
                    "display": "inline",
                    "options": [{
                        "label": "100%",
                        "value": "100%"
                    }, {
                        "label": "95%",
                        "value": "95%"
                    }, {
                        "label": "90%",
                        "value": "90%"
                    }, {
                        "label": "85%",
                        "value": "85%"
                    }, {
                        "label": "80%",
                        "value": "80%"
                    }, {
                        "label": "75%",
                        "value": "75%"
                    }
                    ]
                }, {
                    "type": "dropdown",
                    "label": "Position",
                    "name": "position",
                    "width": 124,
                    "display": "inline",
                    "options": [{
                        "label": "Center",
                        "value": "center"
                    }, {
                        "label": "Left",
                        "value": "left"
                    }, {
                        "label": "Right",
                        "value": "right"
                    }
                    ]
                }
                ]
            }
            ]
        });

        // View JS


        // Edit JS
        module.edit = function() {
            var self = this, 
            websHR = this.websHR = this.el.find('.webs-hr');

            this.toolbar = bldr.toolbar.create('horizontal_rule');
            this.toolbar.addListener({

                'style': function(val) {
                    self.setStyle(val);
                },

                'margin': function(val) {
                    websHR.css({
                        'margin-top': val + 'px',
                        'margin-bottom': val + 'px'
                    });
                    self.data.margin = val;
                },

                'width': function(val) {
                    websHR.css({
                        'width': val
                    });
                    self.data.width = val;
                },

                'position': function(val) {
                    websHR.removeClass('webs-align-' + self.data.position);
                    websHR.addClass('webs-align-' + val);
                    self.data.position = val;
                }

            });
            this.toolbar.show({
                'style': this.getCurrentStyle().slug,
                'margin': this.data.margin,
                'width': this.data.width,
                'position': this.data.position
            });
        };

        module.getData = function() {
            return this.data;
        };

        module.getMinWidth = function() {
            var styleNumber;
            try {
                styleNumber = this.data.className.match(/-(\d+)$/)[1];
            } catch (e) {
                // to be safe, assume the widest divider
                styleNumber = 2;
            }
            var width = 0;
            switch (styleNumber) {
            case "1":
                width = 40;
                break;
            case "2":
                width = 161;
                break;
            case "3":
                width = 59;
                break;
            case "4":
                width = 132;
                break;
            case "5":
                width = 107;
                break;
            case "6":
                width = 83;
                break;
            case "7":
                width = 59;
                break;
            case "8":
                width = 59;
                break;
            case "9":
                width = 67;
                break;
            case "10":
                width = 71;
                break;
            }
            var multiplier = 100 / parseInt(this.data.width);
            return parseInt(width * multiplier);
        };

        module.safeOneLoaded = function() {
            this.chooseDarkOrLightStyle('dark-1', 'light-1');
        };


    })(mod, ext);

    return ModuleClassLoader.register('horizontal_rule', mod, ext);
});




// twitter-feed Module (widget)

// breaks if require is used
define('module/twitter-feed', ['internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.twitter-feed'], function(ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function (module, extend) {

        // Show Cover?
        extend.shouldShowCover = function () {
            return false;
        };

        // Widget
        extend.isWidget = true;



        // Module HTML Template
        extend.template = '{{#if username}}\
        <div class="twitter-feed-resizer" style="height:{{height}}px;">\
        <div class="twitter-feed" id="{{containerId}}"></div>\
        <!--[if IE]>\
        <style type="text/css">\
        /* SITEBUILDER-2009 - In IE, there\'s a bug where if css is written in a style block on the fly and\
        before the rendered element, the css rules won\'t get applied. This is to force styling in IE. Twitter\
        has been notified to fix this */\
        #{{containerId}}.twitter-feed .twtr-doc,\
        #{{containerId}}.twitter-feed .twtr-hd a,\
        #{{containerId}}.twitter-feed h3,\
        #{{containerId}}.twitter-feed h4 {\
        background-color: {{shellBgColor}} !important;\
        color: {{shellTextColor}} !important;\
        }\
        #{{containerId}}.twitter-feed .twtr-tweet a { color: {{linkColor}} !important;}\
        \
        #{{containerId}}.twitter-feed .twtr-bd,\
        #{{containerId}}.twitter-feed .twtr-timeline i a,\
        #{{containerId}}.twitter-feed .twtr-bd p { color: {{tweetTextColor}} !important;}\
        \
        #{{containerId}}.twitter-feed .twtr-new-results,\
        #{{containerId}}.twitter-feed .twtr-results-inner,\
        #{{containerId}}.twitter-feed .twtr-timeline { background: {{tweetBgColor}} !important; }\
        </style>\
        <![endif]-->\
        </div>\
        {{else}}\
        <div class="bldr-placeholder-wrap" style="height:{{height}}px;">\
        <div class="bldr-placeholder-twitter-feed bldr-placeholder-social bldr-placeholder-element">\
        <span class="placeholder-text">{{#l}}webs.module.twitter-feed.placeholder{{/l}}</span>\
        </div>\
        </div>\
        {{/if}}\
        ';

        // Module Default Data
        extend.defaultData = {
            "showAvatars": true,
            "numTweets": 4,
            "shellBgColor": "#ffffff",
            "shellTextColor": "#333333",
            "tweetBgColor": "#ffffff",
            "tweetTextColor": "#333333",
            "linkColor": "#0084B4",
            "theme": "light",
            "username": "",
            "height": "200"
        };

        // Module Styles
        extend.styles = {
            "base": {
                "global": {
                    "css": "view.less"
                },
                "slug": "base"
            }
        };
        if (!extend.styles['base']['global']) {
            extend.styles['base']['global'] = {};
        }
        extend.styles['base']['global']['js'] = null;
        extend.defaultStyle = extend.styles['base'];


        // Register Toolbars
        bldr.toolbar.register('twitter-feed', {
            "icon": "twitter-feed",
            "items": [{
                "type": "textfield",
                "name": "username",
                "label": "Twitter Username",
                "placeholder": "placeholderUsername",
                "width": "200"
            }, {
                "type": "menu",
                "label": "Settings",
                "menuLabel": "Advanced Settings",
                "menuClassNames": "twitter-feed-settings",
                "menu": [{
                    "type": "dropdown",
                    "label": "Text Color",
                    "name": "theme",
                    "width": 60,
                    "display": "inline",
                    "options": [{
                        "label": "Dark",
                        "value": "light"
                    }, {
                        "label": "Light",
                        "value": "dark"
                    }
                    ]
                }, {
                    "display": "inline",
                    "delayUpdate": true,
                    "type": "colorpicker-menu",
                    "label": "Link Color",
                    "name": "linkColor"
                }, {
                    "display": "inline",
                    "delayUpdate": true,
                    "type": "colorpicker-menu",
                    "label": "Background",
                    "name": "shellBgColor"
                }
                ]
            }
            ]
        });

        // View JS
        if (!window.twttr) {
            var t, script, firstScript = document.getElementsByTagName('script')[0];
            script = document.createElement('script');
            script.id = 'twitter-wjs';
            script.src = "https://platform.twitter.com/widgets.js";
            if (document.getElementById(script.id))
                return;
            firstScript.parentNode.insertBefore(script, firstScript);
            window.twttr = window.twttr || (t = {
                _e: [],
                ready: function(f) {
                    t._e.push(f) 
                }
            });
        }

        module.oneLoaded = function() {
            var self = this;
            window.twttr.ready(function() {
                self.display() 
            });
        }

        module.display = function() {
            var data = this.data,
            div = this.el.find(".twitter-feed")
            .empty()
            .css({
                'background-color': data.shellBgColor 
            });

            if (!data.containerId) {
                data.containerId = this.generateId();
                div.attr("id", data.containerId);
                if (this.dirty)
                    this.dirty();
            }

            var lang = webs.locale.split(/[_\-]/)[0];
            if (lang === "zz") {
                lang = "fr";
            }

            data.username = data.username.replace("@", "");

            twttr.widgets.createTimeline(
            '345639413285539841', // This id came from a twitter account owned by Idris
            $('<div/>').appendTo(div)[0],
            {
                screenName: data.username,
                theme: data.theme,
                height: data.height,
                width: data.width,
                linkColor: data.linkColor,
                lang: lang,
                chrome: 'nofooter transparent'
            });
        }

        module.generateId = function() {
            var prefix = "twitter-feed-module-",
            num = Math.floor(Math.random() * (9999999999 - 1000000000 + 1) + 1000000000),
            id = prefix + num;

            return id;
        }

        // Edit JS
        var pgController = bldr.pageController,
        require = pgController.win.require;

        $.extend(module, {
            onActivate: function() {
                var self = this, data = self.data;

                self.toolbar = bldr.toolbar.create('twitter-feed');
                self.toolbar.show($.extend({}, data));
                self.addListeners();
                if (data.username === "webs" || !data.username)
                    this.showHelptip(translate('webs.module.twitter-feed.toolbarHelptip'));
            },
            safeOneLoaded: function() {
                var self = this;
                this.bind("rendered", function() {
                    try {
                        $(document).ready(function() {
                            if (self.data.username) {
                                window.twttr.ready(function() {
                                    self.display() 
                                });
                            }
                        });
                    } catch (e) {
                        webs.log.info(e.message);
                    }
                });
            },
            getData: function() {
                return this.data;
            },
            attachControls: function() {
                this.attachResizer();
            },
            addListeners: function() {
                var self = this;

                self.toolbar.addListener(function(val, name) {
                    webs.log.info(arguments);
                    self.data[name] = val;
                    self.render();
                });
            },
            attachResizer: function() {
                var $el = this.el.find('.twitter-feed-resizer'),
                self = this;

                $el.cropper({
                    reference: $el,
                    sides: ['bottom'],
                    minHeight: 191,
                    maxHeight: 3000,
                    respectMaxDimensionsOnly: true,
                    onMoveEnd: function(side, newWidth, newHeight) {
                        self.data.height = newHeight;
                        self.display();
                    }
                });
            }
        });

    })(mod, ext);

    return ModuleClassLoader.register('twitter-feed', mod, ext);

});


// slideshow Module (custom)

define('module/slideshow', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.slideshow'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="w-slideshow w-slideshow-{{side}} webs-align-{{align}}{{#style}} w-slideshow-{{.}}{{/style}}{{#case images.length 0}} w-slideshow-empty{{/case}} {{#if background}}w-slideshow-bg_enabled{{/if}}" \
        style="width: {{width}}px; height: {{height}}px; {{#if background}}background-color: {{background_color}}{{/if}}">\
        <div class="w-slideshow-thumbnail-area">\
        <div class="w-slideshow-thumbnails">\
        <ul class="w-slideshow-thumbnails-ul">\
        {{#each images}}\
        <li class="w-slideshow-thumbnails-li {{#case _i_ 0}}selected{{/case}}">\
        <figure class="w-slideshow-figure">\
        <div class="w-slideshow-image_container clearfix">\
        <img data-src-large="{{manipulate_image . fitWidth="1920" fitHeight="1920"}}" src="{{manipulate_image . square="250"}}" alt="{{caption}}" class="w-slideshow-image">\
        {{#if caption}}<figcaption class="w-slideshow-caption"><div>{{caption}}</div></figcaption>{{/if}}\
        </div>\
        </figure>\
        </li>\
        {{/each}}\
        {{#case images false}}\
        <li class="w-slideshow-thumbnails-li">\
        <figure class="w-slideshow-figure">\
        <div data-placeholder-text="{{#l}}webs.module.slideshow.doubleClickToAdd{{/l}}" class="w-slideshow-image_container bldr-placeholder-element"></div>\
        </figure>\
        </li>\
        {{/case}}\
        </ul>\
        <nav class="w-slideshow-thumbnails-nav">\
        <a href="#previous_page" title="{{#l}}webs.module.slideshow.previousPage{{/l}}" class="w-slideshow-page-prev"><span>{{#l}}webs.module.slideshow.previousPage{{/l}}</span></a>\
        <a href="#next_page" title="{{#l}}webs.module.slideshow.nextPage{{/l}}" class="w-slideshow-page-next"><span>{{#l}}webs.module.slideshow.nextPage{{/l}}</span></a>\
        </nav>\
        </div>\
        </div>\
        \
        <div class="w-slideshow-resize">\
        <div class="w-slideshow-current">\
        <figure class="w-slideshow-figure">\
        {{#first images}}\
        <div class="w-slideshow-image_container clearfix">\
        <img src="{{url}}" alt="{{caption}}" class="w-slideshow-image">\
        {{#if caption}}<figcaption class="w-slideshow-caption"><div>{{caption}}</div></figcaption>{{/if}}\
        </div>\
        {{^}}\
        <div data-placeholder-text="{{#l}}webs.module.slideshow.doubleClickToAdd{{/l}}" class="w-slideshow-image_container bldr-placeholder-element">\
        <img class="placeholder w-slideshow-image" src="/s/modules/slideshow/v000/images/placeholder.png">\
        </div>\
        {{/first}}\
        </figure>\
        </div>\
        <nav class="w-slideshow-nav">\
        <a href="#previous" title="{{#l}}webs.module.slideshow.previousSlide{{/l}}" class="w-slideshow-prev"><span>{{#l}}webs.module.slideshow.previous{{/l}}</span></a>\
        <a href="#play" title="{{#l}}webs.module.slideshow.play{{/l}}" class="w-slideshow-play"><span>{{#l}}webs.module.slideshow.playSlideshow{{/l}}</span></a>\
        <a href="#next" title="{{#l}}webs.module.slideshow.nextSlide{{/l}}" class="w-slideshow-next"><span>{{#l}}webs.module.slideshow.next{{/l}}</span></a>\
        </nav>\
        </div>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "_style": "white",
            "images": [],
            "align": "center",
            "width": 420,
            "height": 280,
            "thumb_width": 46,
            "thumb_height": 46,
            "autoplay": false,
            "duration": 3000,
            "transition": "fade",
            "side": "bottom",
            "background": false,
            "background_color": "#e1e1e1"
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "base": {
                "global": {
                    "js": "base.view.js"
                },
                "slug": "base"
            },
            "white": {
                "name": "White",
                "inherit": "base",
                "global": {
                    "css": "white.view.less",
                    "js": "white.view.js"
                },
                "default": true,
                "slug": "white"
            }
        };
        extend.styles['base']['global']['js'] = (function(module, extend) {


            /* base.view.js */
            $.extend(module, {
                initBase: function() {
                    if (this.data.side === "none") 
                        return;
                    this.thumbsWidth = this.thumbsUl.outerWidth();
                    this.thumbsLeft = this.thumbsUl.position().left;
                    this.thumbsContainerWidth = this.thumbsContainer.width();
                    this.thumbsContainerOffset = this.thumbsContainer.offset().left;

                    if (this.initStyle) 
                        this.initStyle();
                },
                scrollThumbs: function(e) {
                    if (this.data.side === "none") 
                        return;

                    var
                    midPoint = this.data.width / 2,
                    pos = (e.clientX - this.thumbsContainerOffset),
                    velocity = (pos - midPoint) / midPoint; // percent of mouse position from midpoint (which direction)

                    this.thumbsWidth = this.thumbsUl.outerWidth();
                    this.thumbsLeft = this.thumbsUl.position().left;
                    this.thumbsContainerWidth = this.thumbsContainer.width();
                    this.thumbsContainerOffset = this.thumbsContainer.offset().left;

                    if (this.thumbsWidth > this.thumbsContainerWidth) {
                        var scrollable = this.thumbsContainerWidth - this.thumbsWidth;

                        // Edge cases
                        if (velocity < 0 && this.thumbsLeft >= 0) 
                            this.thumbsUl.css('left', 0);
                        else if (velocity > 0 && this.thumbsLeft <= scrollable) 
                            this.thumbsUl.css('left', scrollable);
                        else {
                            // move slideshow at most .02% of scrollable (if cursor is at edge)
                            this.thumbsUl.css('left', this.thumbsLeft + velocity * (scrollable * 0.02));
                        }
                    }
                }
            });
            return module;
        });
        extend.styles['white']['global']['js'] = (function(module, extend) {


            /* white.view.js */
            $.extend(module, {
                initStyle: function() {
                    /*
                    		this.$timerBar = $("<div/>").addClass("w-slideshow-timerbar").appendTo(this.el.find(".w-slideshow-nav"));
                    		this.$progress = $("<div/>").addClass("w-slideshow-progress").appendTo(this.$timerBar);
                    		this.timerWidth = 76;
                    		*/
                },
                onPlay: function(restart, duration) {
                    //this.$progress.animate({"width": this.timerWidth}, duration);
                },
                onPause: function(stop) {
                    //this.$progress.stop(false, !!stop);
                },
                onReset: function() {
                    //this.$progress.width(0);
                }
            });
            return module;
        });
        extend.defaultStyle = extend.styles['white'];

        // Register Toolbars
        bldr.toolbar.register('slideshow', {
            "icon": "slideshow",
            "items": [{
                "label": "Manage Slideshow",
                "name": "slides",
                "style": "blue",
                "dialog": {
                    "url": "/s/modules/slideshow/v1.29/slideshow.dialog.html",
                    "heading": "Manage Slideshow",
                    "localizeHeading": "webs.module.slideshow.dialog.header",
                    "width": 590,
                    "height": 359
                }
            }, {
                "type": "button-group",
                "sticky": true,
                "distinct": true,
                "items": [{
                    "label": "Thumbnails None",
                    "icon": "thumbs_none",
                    "name": "side",
                    "value": "none"
                }, {
                    "label": "Thumbnails Bottom",
                    "icon": "thumbs_bottom",
                    "name": "side",
                    "value": "bottom"
                }, {
                    "label": "Thumbnails Top",
                    "icon": "thumbs_top",
                    "name": "side",
                    "value": "top"
                }
                ]
            }, {
                "type": "button-group",
                "sticky": true,
                "distinct": true,
                "items": [{
                    "label": "left",
                    "icon": "align_left",
                    "name": "align",
                    "value": "left"
                }, {
                    "label": "center",
                    "icon": "align_center",
                    "name": "align",
                    "value": "center"
                }, {
                    "label": "right",
                    "icon": "align_right",
                    "name": "align",
                    "value": "right"
                }
                ]
            }, {
                "type": "divider"
            }, {
                "type": "menu",
                "label": "Settings",
                "menuLabel": "Slideshow Settings",
                "menuClassNames": "w-slideshow-advanced",
                "menu": [{
                    "display": "inline",
                    "label": "Autoplay",
                    "type": "switch-bare",
                    "name": "autoplay"
                }, {
                    "display": "inline",
                    "label": "Background",
                    "type": "switch-bare",
                    "name": "background"
                }, {
                    "display": "inline",
                    "type": "colorpicker-menu",
                    "label": "Color",
                    "name": "background-color"
                }, {
                    "display": "inline",
                    "label": "Speed",
                    "type": "dropdown",
                    "name": "duration",
                    "width": 100,
                    "options": [{
                        "label": "1 Second",
                        "value": 1000
                    }, {
                        "label": "2 Seconds",
                        "value": 2000
                    }, {
                        "label": "3 Seconds",
                        "value": 3000
                    }, {
                        "label": "4 Seconds",
                        "value": 4000
                    }, {
                        "label": "5 Seconds",
                        "value": 5000
                    }
                    ]
                }, {
                    "display": "inline",
                    "label": "Transition",
                    "type": "dropdown",
                    "name": "transition",
                    "width": 100,
                    "options": [{
                        "label": "None",
                        "value": "none"
                    }, {
                        "label": "Fade",
                        "value": "fade"
                    }, {
                        "label": "Puff",
                        "value": "puff"
                    }, {
                        "label": "H Slide",
                        "value": "slideHorizontal"
                    }, {
                        "label": "V Slide",
                        "value": "slideVertical"
                    }, {
                        "label": "H Carousel",
                        "value": "carouselHorizontal"
                    }, {
                        "label": "V Carousel",
                        "value": "carouselVertical"
                    }
                    ]
                }
                ]
            }
            ]
        });

        // View JS
        require([webs.props.staticServer + "/active-static/target/internal/sitebuilder/modules/common/transitions.js"]);

        /* TODO:
         *
         * Replace every setTimeout we use to get around zero-size image bugs with some browser-normalizing wrapper, like
         *     $img.updateSource(url, callback);
         * That looks something like this:
         *     jquery.fn.updateSource = function(url, callback) {
         *         this.bind("load", function(){ setTimeout(callback, 1)});
         *         this.attr("src", url);
         *         if (this.complete)
         *             setTimeout(callback, 1);
         *     };
         */



        var slideshowModule = {

            /**
            	 * Init function
            	 */
            oneLoaded: function() {

                this.afterLoaded();

            },

            afterLoaded: function() {

                var self = this;
                $(function() {
                    self.preloadImages();
                });

                // Keep these definitons to make sure everything else still works.
                this.currentSlide = this.el.find(".w-slideshow-current");
                this.thumbsContainer = this.el.find(".w-slideshow-thumbnails");
                this.thumbsUl = this.el.find(".w-slideshow-thumbnails-ul");
                this.thumbsLi = this.el.find(".w-slideshow-thumbnails-li");
                this.navContainer = this.el.find(".w-slideshow-nav");
                this.playButton = this.el.find(".w-slideshow-play");

                this.maxWidth = null;
                this.scrollThumbsTimer = 0; // timer for thumbnail hover scrolling

                this.scrollTimer = 0; // timer for the overall slideshow
                this.remainingTime = 0; // stores remaining time during a pause
                this.startTime = 0; // stores start time of slide scroll

                this.transitioning = false; // are we in the middle of a transition?
                this.playing = false; // is the slideshow playing?
                this.slides = []; // array reference to figures

                this.thumbsLi.eq(0).addClass('selected');
                this.loadSlides();
                if (this.initBase) 
                    this.initBase();
                if (this.data.autoplay && this.slides.length > 1) 
                    this.play(true);
                this.ready = true; // bc oneLoaded gets fired after envent binding
                if (this.slides.length <= 1) {
                    this.el.find('.w-slideshow-thumbnails').addClass('no-scroll');
                }

            },

            /**
            	 * DOM events
            	 */
            events: {
                "click .w-slideshow-play": "playClick",
                "click .w-slideshow-prev": "prevClick",
                "click .w-slideshow-next": "nextClick",
                "click .w-slideshow-thumbnails .w-slideshow-image_container": "thumbsClick",
                "mousemove .w-slideshow-thumbnails": "thumbsMousemove",
                "mouseleave .w-slideshow-thumbnails": "thumbsMouseleave"
            },

            prevClick: function(e) {
                if (this.ready)
                    this.prev();
                e.preventDefault();
                return false;
            },
            nextClick: function(e) {
                if (this.ready)
                    this.next();
                e.preventDefault();
                return false;
            },
            playClick: function(e) {
                if (this.ready)
                    this[this.scrollTimer > 0 ? "pause" : "play"]();
                e.preventDefault();
                return false;
            },

            thumbsClick: function(e) {
                if (this.ready) {
                    this.stop();
                    this.select($(e.target).parents(".w-slideshow-figure"));
                }
            },
            thumbsMouseleave: function() {
                if (this.ready) {
                    clearInterval(this.scrollThumbsTimer);
                    delete this.scrollThumbsTimer;
                }
            },
            thumbsMousemove: function(e) {
                if (this.ready) {
                    var self = this;
                    if (this.scrollThumbsTimer) 
                        clearInterval(this.scrollThumbsTimer);
                    this.scrollThumbsTimer = setInterval(function() {
                        self.scrollThumbs(e);
                    }, 10);
                }
            },

            loadSlides: function() {
                var self = this;
                if (this.data.images.length === 0) {
                    this.resizePlaceholder();
                    return;
                }
                this.slides = [];
                this.thumbsLi.each(function(i, li) {
                    var
                    $fig = i === 0 ? self.currentSlide.children() : $(li).children().clone(),
                    $img = $fig.find("img");

                    self.getImageDimension($img.data("srcLarge") || $img.attr("src"), function(w, h) {
                        self.postloadProcess($img, w, h);
                    });
                    if ($img.data("srcLarge")) {
                        $img.attr("src", $img.data("srcLarge"));
                    }

                    self.slides[i] = $fig.data("img", $img);
                });
            },

            resizePlaceholder: function() {
                var $elm = this.el.find(".w-slideshow-resize .bldr-placeholder-element"),
                height = $elm.parent().height(),
                padding = $elm.outerHeight() - $elm.height(),
                self = this;
                if (height < 1) 
                    return setTimeout(function() {
                        self.resizePlaceholder();
                    }, 1000);
                $elm.css({
                    width: "auto",
                    height: (height - padding) + "px"
                });
            },

            getImageDimension: function(src, fn) {
                var image = $("<img/>");
                image.load(function() {
                    if (typeof fn === "function") 
                        fn(this.width, this.height);
                }).attr("src", src);
                if (image[0].complete && typeof fn === "function") {
                    setTimeout(function() {
                        fn(image[0].width, image[0].height);
                    }, 1);
                }
            },

            postloadProcess: function($img, _width, _height) {
                var _ratio = _width / _height;

                var paddingNode = this.currentSlide.find(".w-slideshow-image_container");
                var widthPadding = paddingNode.outerWidth() - paddingNode.width();
                var heightPadding = paddingNode.outerHeight() - paddingNode.height();
                if (!this.maxWidth) {
                    this.maxWidth = this.data.width - widthPadding;
                    this.maxHeight = this.data.height - heightPadding;
                }
                var slideRatio = this.maxWidth / this.maxHeight;

                $img.data("width", _width).data("height", _height).data("ratio", _ratio);

                var cssObj = {};
                if (_width < this.maxWidth && _height < this.maxHeight) {
                    cssObj.display = "inline-block";
                    cssObj.top = parseInt((this.maxHeight - _height) / 2, 10);
                } else if (slideRatio > _ratio) {
                    cssObj.display = "block";
                    cssObj.height = this.maxHeight;
                    cssObj.width = parseInt(this.maxHeight * _ratio, 10);
                } else {
                    cssObj.display = "block";
                    cssObj.width = "auto"; //this.maxWidth;
                    cssObj.height = "auto"; //parseInt(this.maxWidth/_ratio, 10);
                    cssObj.top = parseInt((this.maxHeight - parseInt(this.maxWidth / _ratio, 10)) / 2, 10);
                }
                $img.parent().css(cssObj);
            },

            scrollThumbs: function() {},

            select: function(fig, callback) {
                var self = this, newThumb = fig.parent();

                callback = callback || function() {};

                if (this.playing) 
                    this.stop(); // Stop the slideshow (and set the timer to 0).
                if (this.transitioning || newThumb.hasClass('selected')) {
                    callback();
                    return false;
                }

                this.currentSlide = this.el.find(".w-slideshow-current");


                var
                oldThumb = this.el.find('.w-slideshow-thumbnails .selected'),
                oldI = oldThumb.index(),
                oldFig = this.slides[oldI],
                slen = oldThumb.siblings().length,
                newI = newThumb.index(),
                newFig = this.slides[newI].appendTo(this.currentSlide),
                // grab transition information
                transition = webs.transitions[this.data.transition] || webs.transitions.none,
                invertTransition = (newI < oldI && !(newI === 0 && oldI === slen)) || (newI === slen && oldI === 0);

                // setting class for transition and state
                oldThumb.removeClass("selected");
                oldFig.addClass("outgoing");
                newThumb.addClass("selected");
                newFig.addClass("incoming");

                this.transitioning = true;
                transition(oldFig, newFig, 400, invertTransition, function() {
                    self.transitioning = false;
                    oldFig.detach().removeClass("outgoing");
                    newFig.removeClass('incoming').css('z-index', '');
                    callback();
                });

                // make sure thumbs are in view, unless the user is scrolling through the thumbs
                if (!this.scrollThumbsTimer) {
                    var figLI = fig.parent(),
                    width = this.thumbsContainer.width(),
                    edgeLeft = this.thumbsContainer.offset().left,
                    edgeRight = edgeLeft + width,
                    figLeft = figLI.offset().left,
                    figRight = figLeft + figLI.outerWidth();

                    if (figRight > edgeRight) {
                        this.thumbsUl.animate({
                            'left': '-=' + (figRight - edgeRight) 
                        }, 100);
                    } else if (figLeft < edgeLeft) {
                        this.thumbsUl.animate({
                            'left': '+=' + (edgeLeft - figLeft) 
                        }, 100);
                    }
                }
            },

            prev: function(cb) {
                var fig = this.el.find('.selected').prev().children(".w-slideshow-figure");
                if (fig.length === 0) {
                    fig = this.el.find('.w-slideshow-thumbnails li:last-child .w-slideshow-figure');
                }
                this.select(fig, cb);
            },
            next: function(cb) {
                var fig = this.el.find('.selected').next().children(".w-slideshow-figure");
                if (fig.length === 0) {
                    fig = this.el.find('.w-slideshow-thumbnails li:first-child .w-slideshow-figure');
                }
                this.select(fig, cb);
            },
            play: function(restart) {
                var self = this;

                this.el.find('.w-slideshow-play').addClass('w-slideshow-pause').attr('title', translate('webs.module.slideshow.pause'));
                this.playing = true;

                // If coming from a pause, remainingTime should be > 0, use this as duration
                this.startTime = new Date();
                this.remainingTime = this.remainingTime > 0 ? this.remainingTime : this.data.duration;

                this.scrollTimer = setTimeout(function() {
                    var wasPlaying = self.playing;
                    self.playing = false; // Set playing to false so our 'select' logic doesn't pause the slideshow.
                    self.remainingTime = 0;
                    self.next(function() {
                        if (wasPlaying) 
                            self.play(true);
                    });
                }, this.remainingTime);

                if (restart) 
                    this.reset();
                if (this.onPlay) 
                    this.onPlay(restart, this.remainingTime);
            },
            pause: function(stop) {
                this.el.find('.w-slideshow-play').removeClass('w-slideshow-pause').attr('title', translate('webs.module.slideshow.play'));
                this.playing = false;

                // When pausing, store the remaining time in the queue
                clearTimeout(this.scrollTimer);
                this.scrollTimer = 0;
                this.remainingTime -= new Date() - this.startTime;

                if (stop) {
                    this.remainingTime = 0;
                    this.reset();
                }
                if (this.onPause) 
                    this.onPause(stop);
            },
            stop: function() {
                return this.pause(true);
            },

            reset: function() {
                if (this.onReset) 
                    this.onReset();
            },

            preloadImages: function() {
                this.el.find('.w-slideshow-thumbnails-li img').each(function() {
                    new Image().src = $(this).data('src-large');
                });
            }
        };

        var mobileSlideshow = {
            events: {
                "touchstart .w-slideshow-thumbnails ul": "stopMobile"
            },

            oneLoaded: function() {

                this.afterLoaded();

                // Center images based on their height if they are too tall.
                var parentHeight = $(".w-slideshow-thumbnails").height();
                $(".w-slideshow-image").each(function() {
                    var self = $(this);
                    if (self.height() > parentHeight) {
                        self.css({
                            "top": (parentHeight - self.height()) / 2
                        });
                    }
                });

                // don't let scroll events from the slideshow bubble up to the viewport, which will cause the entire viewport to move
                $('.w-slideshow-thumbnails').on('touchmove mousemove', function(e) {
                    e.stopPropagation();
                });

            },

            stopMobile: function() {
                this.stop();
                this.el.find('.w-slideshow-thumbnails ul').off('touchstart');
            },

            next: function(cb) {
                var current = this.el.find('.selected');
                var next;
                current.removeClass("selected");
                if (current.next().length === 0) {
                    next = this.el.find('.w-slideshow-thumbnails li:first-child');
                    next.addClass("selected");
                } else {
                    next = current.next();
                }
                next.addClass("selected");
                this.thumbsContainer.animate({
                    scrollLeft: next.position().left - 20
                }, 500);
                cb();
            }
        };

        $.extend(module, slideshowModule);

        // Extend for mobile if on mobile theme.
        if (webs && webs.theme && webs.theme.mobile) {
            $.extend(module, mobileSlideshow);
        }


        // Edit JS
        /**
        * Initialize edit mode
        */
        module.onActivate = function() {
            if (this.data.autoplay && this.slides.length > 1) 
                this.play(true);
        };
        module.safeOneLoaded = function() {
            var self = this;

            this.bind("rendered", function() {
                if (self.scrollThumbsTimer) {
                    clearInterval(self.scrollThumbsTimer);
                    selfscrollThumbsTimer = null;
                }
                self.oneLoaded();

                // in edit mode, only autoplay when activating
                self.stop();
            });
        };

        module.showSettingsDialog = function() {
            this.toolbar.click("slides");
        };

        var $sideTT = null,
        $bodyTT = null;

        module.attachControls = function() {

            var self = this, $ = bldr.pageController.$, data = self.data, container = self.el.find(".w-slideshow");
            var styles = this.getStyles();

            if (!$sideTT)
                $sideTT = bldr.pageController.$.tooltip({
                    style: "warn",
                    content: translate('webs.module.slideshow.warning.sidebarMaxWidth')
                });
            if (!$bodyTT)
                $bodyTT = bldr.pageController.$.tooltip({
                    style: "warn",
                    content: translate('webs.module.slideshow.warning.containerMaxWidth')
                });

            self.toolbar = bldr.toolbar.create('slideshow');

            // update the toolbar with the correct dropdown items
            self.toolbar.updateUI({
                style: styles 
            });

            self.toolbar.show({
                'slides': data.images,
                'align': data.align,
                'side': data.side,
                'transition': data.transition,
                'duration': data.duration,
                'style': data._style,
                'autoplay': data.autoplay,
                'background': data.background,
                'background-color': data['background_color']
            });

            container.find(".w-slideshow-image_container, .bldr-placeholder-element").live("dblclick", function() {
                self.toolbar.click('slides');
            });

            self.toolbar.feedback = {
                ref : self.el.find('.w-slideshow') 
            };
            self.toolbar.addListener({
                'style': function(slug) {
                    self.setStyle(slug);
                },
                'autoplay': function(val) {
                    if (typeof val === "string") {
                        if (val === "true") 
                            val = true;
                        else if (val === "false") 
                            val = false;
                        else 
                            return false;
                    }
                    data.autoplay = val;
                },
                'duration': function(duration) {
                    self.data.duration = parseInt(duration, 10);
                },
                'transition': function(transition) {
                    self.data.transition = transition;
                    // transition to the next one, as a demo of what they just chose
                    self.next();
                },
                'align': function(alignment) {
                    var el = self.el.find('.w-slideshow');
                    el.removeClass('webs-align-' + self.data.align);
                    self.data.align = alignment;
                    el.addClass('webs-align-' + alignment);
                },
                'side': function(side) {
                    var old = self.data.side;
                    self.data.side = side;
                    /*
                    			self.el.find('.w-slideshow')
                    				.removeClass('w-slideshow-' + old)
                    				.addClass('w-slideshow-' + side);
                    				*/
                    self.render();
                    self.oneLoaded();
                },
                'slides': function(data) {
                    self.data.images = data;
                    setTimeout(function() {
                        self.render();
                        self.oneLoaded();
                    }, 300); // 300ms for websover feedback animation
                },
                'background': function(val) {
                    if (typeof val === "string") {
                        if (val === "true") 
                            val = true;
                        else if (val === "false") 
                            val = false;
                        else 
                            return false;
                    }
                    data.background = val;
                    container[val ? 'addClass' : 'removeClass']('w-slideshow-bg_enabled');
                    container.css('background-color', val ? data['background_color'] : '');
                },
                'background-color': function(val) {
                    self.toolbar.show({
                        background: true
                    });
                    data.background = true;
                    data['background_color'] = val;
                    container.css("background-color", val);
                }
            });


            /* Editing interface stuff */
            container.resizer({
                reference: container,
                corners: ['right', 'left'],
                centered: self.data.align === 'center',
                maxWidth: self.el.width(),
                lockRatio: false,
                minWidth: self.minWidth,
                minHeight: self.minHeight,
                onStart: function() {
                    container.addClass("w-slideshow-resizing");
                    self.stop();
                },
                onMove: function(corner, newWidth, newHeight) {
                    if (newWidth <= self.minWidth) 
                        newWidth = self.minWidth;
                    if (newHeight < self.minHeight) 
                        newHeight = self.minHeight;

                    self.data.width = parseInt(newWidth, 10);
                    self.data.height = parseInt(newHeight, 10);
                    container.width(newWidth).height(newHeight);
                    if (self.data.width == self.el.width()) {
                        if (self.el.parents(".webs-sidebar").length)
                            $sideTT.addClass("active");
                        else
                            $bodyTT.addClass("active");
                    } else {
                        $sideTT.removeClass("active");
                        $bodyTT.removeClass("active");
                    }
                    self.redraw(container.find(".w-slideshow-resize"));
                },
                onEnd: function() {
                    container.removeClass("w-slideshow-resizing");
                    $sideTT.removeClass("active");
                    $bodyTT.removeClass("active");
                    self.render();
                    self.oneLoaded();
                }
            });
        };

        module.minWidth = 150;
        module.minHeight = 160;

        module.fit = function(width, persist) {
            if (width < this.data.width) {
                var aspect_ratio = this.data.width / this.data.height,
                height = width / aspect_ratio;
                if (!isFinite(height)) {
                    height = width;
                }
                this.el.find('.w-slideshow').width(width);
                if (persist) {
                    this.data.width = width;
                    this.data.height = height;
                    this.dirty();
                    this.render();
                }
            }
        };

        module.getMinWidth = function() {
            return this.minWidth;
        };

        module.getData = function() {
            return this.data;
        };

        module.onDeactivate = function() {
            this.stop();
        };


    })(mod, ext);

    return ModuleClassLoader.register('slideshow', mod, ext);
});




// follow Module (widget)

// breaks if require is used
define('module/follow', ['internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.follow'], function(ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function (module, extend) {

        // Show Cover?
        extend.shouldShowCover = function () {
            return false;
        };

        // Widget
        extend.isWidget = true;



        // Module HTML Template
        extend.template = '{{#if username}}\
        <div class="webs-align-{{align}}">\
        <a href="https://twitter.com/{{username}}" class="twitter-follow-button" data-show-count="{{#if showFollow}}true{{else}}false{{/if}}"{{#if buttonSize}} data-size="{{buttonSize}}"{{/if}} {{#if align}} data-align="{{align}}"{{/if}} data-lang="{{lang}}">{{#l}}webs.module.follow.Follow{{/l}} @{{username}}</a>\
        <script>!function(d,s,id){var js,fjs=d.getElementsByTagName(s)[0];if(1){js=d.createElement(s);js.id=id;js.src="//platform.twitter.com/widgets.js";fjs.parentNode.insertBefore(js,fjs);}}(document,"script","twitter-wjs");</script>\
        </div>\
        {{else}}\
        <div class="bldr-placeholder-wrap" style="height:200px;">\
        <div data-placeholder-text="{{#l}}webs.module.follow.clickToEdit{{/l}}" class="bldr-placeholder-follow bldr-placeholder-social bldr-placeholder-element">\
        </div>\
        </div>\
        {{/if}}\
        ';

        // Module Default Data
        extend.defaultData = {
            "showFollow": true,
            "buttonSize": "medium",
            "username": "",
            "align": "left",
            "lang": "en"
        };

        // Module Styles
        extend.styles = {
            "base": {
                "global": {
                    "css": "view.less"
                },
                "slug": "base"
            }
        };
        if (!extend.styles['base']['global']) {
            extend.styles['base']['global'] = {};
        }
        extend.styles['base']['global']['js'] = null;
        extend.defaultStyle = extend.styles['base'];


        // Register Toolbars
        bldr.toolbar.register('follow', {
            "icon": "follow",
            "items": [{
                "type": "textfield",
                "name": "username",
                "label": "Twitter Username",
                "placeholder": "placeholderUsername",
                "width": "200"
            }, {
                "type": "button-group",
                "sticky": true,
                "distinct": true,
                "items": [{
                    "label": "Left",
                    "icon": "align_left",
                    "name": "align",
                    "value": "left"
                }, {
                    "label": "Center",
                    "icon": "align_center",
                    "name": "align",
                    "value": "center"
                }, {
                    "label": "Right",
                    "icon": "align_right",
                    "name": "align",
                    "value": "right"
                }
                ]
            }, {
                "type": "menu",
                "label": "Settings",
                "menuLabel": "Advanced Settings",
                "width": 225,
                "menu": [{
                    "display": "inline",
                    "label": "Follower Count",
                    "type": "switch-bare",
                    "name": "showFollow"
                }, {
                    "type": "dropdown",
                    "label": "Button Size",
                    "name": "buttonSize",
                    "width": 80,
                    "display": "inline",
                    "options": [{
                        "label": "Medium",
                        "value": "medium"
                    }, {
                        "label": "Large",
                        "value": "large"
                    }
                    ]
                }
                ]
            }
            ]
        });

        // View JS
        if (!extend.defaultData) {
            extend.defaultData = {
                "lang": "en"
            };
        }

        if (typeof(webs) !== 'undefined' && webs.locale) {
            var language = webs.locale.split(/[-_]/)[0];

            if (language === "zz") {
                language = "fr";
            }

            extend.defaultData.lang = language;

        }

        // Edit JS
        module.safeOneLoaded = function() {
            this.helptip = {
                "fieldName": "username",
                "message": translate("webs.module.follow.helptip")
            };
        };

    })(mod, ext);

    return ModuleClassLoader.register('follow', mod, ext);

});


// header_editor Module (custom)

define('module/header_editor', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.header_editor'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return true;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '\
        <div class="w-header-area" style="width: {{width}};">\
        {{#if showBackground}}\
        <style>\
        {{#background}} .webs-header-bg { background-image: url(\'{{url}}\'); background-position: {{#position}}{{left}}px {{top}}px{{/position}}; } {{/background}}\
        </style>\
        {{/if}}\
        \
        {{#each text_layers}}\
        <div id="{{id}}" class="w-header-layer w-header-layer-text w-header-layer-align-{{align}} {{#if center}}w-header-layer-centered{{/if}}" style="\
        {{#unless center}}left:{{left}}px; right:{{right}}px; {{/unless}}\
        top:{{top}}px; z-index: {{z-index}};">\
        <div class="w-header-layer-container">\
        <div class="w-header-text-container">\
        <div class="w-header-text {{#if font}}w-font-{{font}}{{/if}}" data-placeholder="{{#l}}webs.module.header_editor.placeholder.text{{/l}}">\
        {{#if text}}\
        {{{text}}}\
        {{else}}\
        {{#l}}webs.module.header_editor.placeholder.text{{/l}}\
        {{/if}}\
        </div>\
        </div>\
        </div>\
        </div>\
        {{/each}}\
        \
        \
        {{#each image_layers}}\
        {{#wrapInFrame}}\
        <div id="{{id}}" class="w-header-layer w-header-layer-image" style="left:{{left}}px; top:{{top}}px; z-index: {{z-index}}; width:{{width}}px; height:{{height}}px;">\
        <div class="w-header-layer-container {{#unless url}}w-header-image-placeholder{{/unless}}">\
        {{#if url}}\
        <img {{#if alt}}alt="{{alt}}"{{/if}} class="w-header-image" src="{{manipulate_image . fitWidth="1920" fitHeight="1920"}}"/>\
        {{else}}\
        <div class="placeholder">\
        <div class="w-header-placeholder-icon"></div>\
        <p>{{#l}}webs.module.header_editor.placeholder.image{{/l}}</p>\
        </div>\
        <img class="w-header-image"/>\
        {{/if}}\
        </div>\
        </div>\
        {{/wrapInFrame}}\
        {{/each}}\
        \
        {{#logo}}\
        <div id="{{id}}" class="w-header-layer w-header-layer-image w-header-layer-logo" style="left:{{left}}px; top:{{top}}px; width:{{width}}px; height:{{height}}px; display:{{display}};">\
        <div class="w-header-layer-container {{#unless url}}w-header-logo-placeholder{{/unless}}">\
        {{#if url}}\
        <img class="w-header-image w-header-logo" src="{{manipulate_image . fitWidth="1920" fitHeight="1920"}}"/>\
        {{else}}\
        <div class="placeholder">\
        <div class="w-header-placeholder-icon"></div>\
        <p>{{#l}}webs.module.header_editor.placeholder.image{{/l}}</p>\
        </div>\
        <img class="w-header-image w-header-logo"/>\
        {{/if}}\
        </div>\
        </div>\
        {{/logo}}\
        \
        {{#title}}\
        <div id="{{id}}" class="title w-header-layer w-header-layer-text w-header-layer-align-{{align}} {{#if center}}w-header-layer-centered{{/if}}" style="\
        {{#unless center}}left:{{left}}px; right:{{right}}px; {{/unless}}\
        top:{{top}}px; z-index: {{z-index}}; display: {{display}}">\
        <div class="w-header-layer-container">\
        <h1 class="w-header-text-container">\
        <div class="w-header-text w-header-title {{#if font}}w-font-{{font}}{{/if}}" data-placeholder="{{#l}}webs.module.header_editor.placeholder.title{{/l}}">\
        {{#if text}}\
        {{{text}}}\
        {{else}}\
        {{#l}}webs.module.header_editor.placeholder.title{{/l}}\
        {{/if}}\
        </div>\
        </h1>\
        </div>\
        </div>\
        {{/title}}\
        \
        {{#subtitle}}\
        <div id="{{id}}" class="subtitle w-header-layer w-header-layer-text w-header-layer-align-{{align}} {{#if center}}w-header-layer-centered{{/if}}" style="\
        {{#unless center}}left:{{left}}px; right:{{right}}px; {{/unless}}\
        top:{{top}}px; z-index: {{z-index}}; display: {{display}}">\
        <div class="w-header-layer-container">\
        <h2 class="w-header-text-container">\
        <div class="w-header-text w-header-subtitle {{#if font}}w-font-{{font}}{{/if}}" data-placeholder="{{#l}}webs.module.header_editor.placeholder.subtitle{{/l}}">\
        {{#if text}}\
        {{{text}}}\
        {{else}}\
        {{#l}}webs.module.header_editor.placeholder.subtitle{{/l}}\
        {{/if}}\
        </div>\
        </h2>\
        </div>\
        </div>\
        {{/subtitle}}\
        \
        </div><!-- END w-header-area -->\
        ';

        // Module Default Data
        extend.defaultData = {
            "width": "100%",
            "linksData": {},
            "text_layers": [],
            "image_layers": [],
            "showBackground": false,
            "background": {
                "url": "",
                "position": {
                    "left": 0,
                    "top": 0
                },
                "mobile": {
                    "custom": false,
                    "color": null
                }
            },
            "title": {
                "text": null,
                "top": 22,
                "left": 4,
                "right": "auto",
                "align": "left",
                "z-index": 20,
                "display": "block",
                "mobile": {
                    "display": "block",
                    "top": 18,
                    "left": null,
                    "color": null
                }
            },
            "subtitle": {
                "text": null,
                "top": 42,
                "left": 4,
                "right": "auto",
                "align": "left",
                "z-index": 21,
                "display": "none",
                "mobile": {
                    "display": "none",
                    "top": 40,
                    "left": null,
                    "color": null
                }
            },
            "logo": {
                "top": 16,
                "left": 16,
                "display": "none",
                "width": 74,
                "height": 73,
                "mobile": {
                    "display": "none",
                    "top": 10,
                    "left": null,
                    "color": null
                }
            }
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "instance": {
                    "css": "view.each.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars
        bldr.toolbar.register('header_editor', {
            "icon": "default",
            "items": [{
                "type": "text",
                "text": "Click on an element to show edit options"
            }
            ]
        });
        bldr.toolbar.register('header_layer_text', {
            "icon": "text",
            "items": [{
                "type": "text",
                "text": "Click on your text to edit it"
            }
            ]
        });
        bldr.toolbar.register('header_layer_image', {
            "icon": "image",
            "items": [{
                "type": "button",
                "style": "blue",
                "label": "Change Image",
                "name": "image",
                "dialog": {
                    "url": "/s/sitebuilder/dialogs/image",
                    "localizeHeading": "webs.module.header_editor.changeImage",
                    "width": 650,
                    "height": 361
                }
            }, {
                "icon": "border",
                "type": "framesets",
                "label": "Frame",
                "name": "frame",
                "width": 170
            }, {
                "type": "button-cancel",
                "icon": "link",
                "showLabel": true,
                "label": "Link",
                "name": "imageLink",
                "groupClass": "w-header_editor-linkgroup",
                "classNames": "w-header_editor-link",
                "dialog": {
                    "url": "/s/sitebuilder/dialogs/link",
                    "localizeHeading": "webs.bldr.dialog.link.popover.heading",
                    "width": 590,
                    "height": 361
                },
                "cancel": {
                    "style": "red",
                    "icon": "unlink",
                    "label": "Remove Link",
                    "name": "removeLink"
                }
            }, {
                "type": "button-group",
                "items": [{
                    "type": "button",
                    "label": "Layer Up",
                    "icon": "layerup",
                    "name": "layerUp",
                    "event": "mousedown"
                }, {
                    "type": "button",
                    "label": "Layer Down",
                    "icon": "layerdown",
                    "name": "layerDown",
                    "event": "mousedown"
                }
                ]
            }
            ]
        });
        bldr.toolbar.register('header_layer_logo', {
            "icon": "image",
            "items": [{
                "type": "button",
                "style": "blue",
                "label": "Change Logo",
                "name": "image",
                "dialog": {
                    "url": "/s/sitebuilder/dialogs/image",
                    "localizeHeading": "webs.module.header_editor.changeLogo",
                    "width": 650,
                    "height": 361
                }
            }
            ]
        });
        bldr.toolbar.register('header_text', {
            "icon": "text",
            "items": [{
                "type": "dropdown_fonts",
                "name": "SetFont"
            }, {
                "type": "dropdown_font_size",
                "width": 50,
                "selectedValue": 14,
                "name": "wzSetFontSize",
                "dataType": "text",
                "options": [{
                    "label": "fontSize.8",
                    "value": "8"
                }, {
                    "label": "fontSize.10",
                    "value": "10"
                }, {
                    "label": "fontSize.11",
                    "value": "11"
                }, {
                    "label": "fontSize.12",
                    "value": "12"
                }, {
                    "label": "fontSize.14",
                    "value": "14"
                }, {
                    "label": "fontSize.16",
                    "value": "16"
                }, {
                    "label": "fontSize.18",
                    "value": "18"
                }, {
                    "label": "fontSize.24",
                    "value": "24"
                }, {
                    "label": "fontSize.36",
                    "value": "36"
                }, {
                    "label": "fontSize.48",
                    "value": "48"
                }, {
                    "label": "fontSize.72",
                    "value": "72"
                }
                ]
            }, {
                "type": "colorpicker",
                "icon": "text_color",
                "label": "Color",
                "menuLabel": "Text Color",
                "name": "wzForeColor",
                "value": "#069FF0"
            }, {
                "type": "button-group",
                "sticky": true,
                "distinct": false,
                "items": [{
                    "label": "Bold",
                    "icon": "bold",
                    "name": "bold"
                }, {
                    "label": "Italic",
                    "icon": "italic",
                    "name": "italic"
                }, {
                    "label": "Underline",
                    "icon": "underline",
                    "name": "underline"
                }, {
                    "label": "Strikethrough",
                    "icon": "strike",
                    "name": "strikethrough"
                }
                ]
            }, {
                "type": "button-group",
                "sticky": true,
                "distinct": true,
                "items": [{
                    "label": "Align Left",
                    "icon": "align_left",
                    "name": "Alignment",
                    "value": "left"
                }, {
                    "label": "Align Center",
                    "icon": "align_center",
                    "name": "Alignment",
                    "value": "center"
                }, {
                    "label": "Align Right",
                    "icon": "align_right",
                    "name": "Alignment",
                    "value": "right"
                }
                ]
            }, {
                "type": "button",
                "icon": "eraser",
                "label": "Clear Formatting",
                "name": "RemoveFormat"
            }, {
                "type": "button-cancel",
                "icon": "link",
                "showLabel": true,
                "label": "Link",
                "name": "wzLink",
                "groupClass": "w-wizzy-linkgroup",
                "classNames": "w-wizzy-linkbutton",
                "dialog": {
                    "url": "/s/sitebuilder/dialogs/link",
                    "localizeHeading": "webs.bldr.dialog.link.popover.heading",
                    "width": 590,
                    "height": 366
                },
                "cancel": {
                    "style": "red",
                    "icon": "unlink",
                    "label": "Remove Link",
                    "name": "wzUnlink"
                }
            }
            ]
        });

        // View JS
        var headerEditor = {
            oneLoaded: function() {}
        };

        // On mobile, we want to replace any links with spans
        var mobileHeaderEditor = {
            oneLoaded: function() {
                var textLayers = $('.w-header-layer-text'),
                anchors = textLayers.find('a[href]');

                anchors.each(function() {
                    $(this).replaceWith('<span>' + this.innerHTML + '</span>');
                });
            }
        }

        $.extend(module, headerEditor);

        // Extend for mobile if on mobile theme.
        if (webs && webs.theme && webs.theme.mobile) {
            $.extend(module, mobileHeaderEditor);
        }

        // Edit JS

        //this.controls.hide();
        var CLASSES = {

            LAYER : 'w-header-layer',
            ACTIVELAYER : 'w-header-layer-active',
            INACTIVELAYER : 'w-header-layer-inactive',
            BUTTONDISABLED : 'webs-btn_disabled',
            GRIDON : 'w-header-gridon'
        };

        // z-index ranges for different layer types
        var ZRANGE = {
            image: [0, 10],
            text: [10, 20] 
        };

        // Tags constants for layer types. Not using img bc that's the differentiator
        var HTMLTAG = {
            text: '<div/>',
            title: '<h1/>',
            subtitle: '<h2/>' 
        };

        var PLACEHOLDERTEXT = {
            text: translate('webs.module.header_editor.placeholder.text'),
            title: translate('webs.module.header_editor.placeholder.title'),
            subtitle: translate('webs.module.header_editor.placeholder.subtitle')
        };

        // Utility to hide tooltip, since we do this multiple places
        function hideImageLayerTooltip() {
            bldr.pageController.$("body").trigger("hideTooltip");
        }


        /**
         * Base Layer Controller, describes a layer on the canvas
         */
        var LayerController = {};
        LayerController['base'] = Spine.Controller.create({
            init: function() {
                this.fixData();
                this.active = false;

                var btn = this.parent.fixedButtons[this.type];
                this.button = btn.hasClass("w-switch") ? btn : btn.children();

                if (this.onInitBase) 
                    this.onInitBase();
                this.render();
            },

            fixData: function() {
                if (this.data.top < 0) 
                    this.data.top = 0;
            },

            render: function() {
                var self = this;

                // Grab frame and el reference
                if (this.frame) 
                    this.frame.remove();
                this.frame = this.$(this.tmpl(this.data)).appendTo(this.parent.$canvas);
                this.el = this.frame.hasClass("webs-frame") ? this.frame.find(".w-header-layer") : this.frame;
                this.container = this.el.find(".w-header-layer-container");

                // Existing sites may be broken; fix these so users can get their drag handles.
                if (this.data.top < 0) 
                    this.data.top = 0;

                this.el.data("controller", this).addClass(CLASSES.LAYER + ' ' + CLASSES.LAYER + '-' + this.type);
                this.frame.css({
                    'left': this.data.left,
                    'right': this.data.right,
                    'top': this.data.top
                });

                this.titlebar = this.$("<div/>").addClass("w-header-titlebar").appendTo(this.container);

                this.moveButton = this.$("<span/>")
                .appendTo(this.titlebar)
                .attr({
                    'class': 'w-header-layer-move',
                    'title': translate('webs.module.header_editor.movelayer')
                })
                .html(translate('webs.module.header_editor.movelayer'))
                .mousedown(function() {
                    return self.moveButtonMousedown();
                });

                this.delButton = this.$('<a/>')
                .appendTo(this.titlebar)
                .attr({
                    'class': 'w-header-layer-delete',
                    'title': translate('webs.module.header_editor.deletelayer') 
                })
                .html(translate('webs.module.header_editor.deletelayer'))
                .mousedown(function() {
                    return self.deleteButtonMousedown();
                });

                if (this.onRender) 
                    this.onRender();

                this.attachEvents();
                this.attachDragger();
            },

            moveButtonMousedown: function() {
                this.activate();
                return false;
            },

            deleteButtonMousedown: function() {
                hideImageLayerTooltip();
                this.remove();
                this.parent.hideSettingsMenu();
                return false;
            },

            // We have a manual attachEvents method that works off the _events object,
            // Since there's a possibility that we'll rerender and need to rebind events
            attachEvents: function() {
                var e = this._events || [];
                for (var i = 0, len = e.length; i < len; i++) {
                    this._attachEvent(e[i][0], e[i][1], e[i][2]);
                }
                if (this.onAttachEvents) 
                    this.onAttachEvents();
            },
            _attachEvent: function(evt, sel, key) {
                var self = this, el = sel === "" ? this.frame : this.frame.find(sel);
                el[evt](function(e) {
                    return self[key](e);
                });
            },

            // Activates module, will deactivate a current active module first
            activate: function() {
                var module = this.parent;

                // don't activate if already active;
                if (module.activeLayer && this === module.activeLayer) 
                    return false;
                module.deactivateLayer(true);
                module.activateLayer(this);

                this.active = true;
                this.frame.addClass(CLASSES.ACTIVELAYER);
                if (this.onActivate) 
                    this.onActivate();
            },

            // Deactivate layer, called from module
            deactivate: function(fromActivate) {
                this.active = false;
                this.frame.removeClass(CLASSES.ACTIVELAYER);
                if (this.onDeactivate) 
                    this.onDeactivate();

                // Always showing toolbar might conflict with toolbar if coming from layer activate
                if (!fromActivate) 
                    this.parent.toolbar.header.show();
            },

            destroy: function() {
                try {
                    if (this.frame.dragger) 
                        this.frame.dragger("destroy");
                    if (this.onDestroy) 
                        this.onDestroy();
                } catch (err) {}
            },

            // Checks to see if layer is ancestor of an element
            contains: function(el) {
                return this.el.find(el).length;
            },

            // Attach dragger controls to layer
            attachDragger: function() {
                var
                self = this,
                module = this.parent,
                canvasWidth = module.dimension.width,
                canvasHeight = module.dimension.height,
                data = this.data,
                frame = this.frame,
                layer = this.el;

                frame.dragger({
                    apply: false,

                    ref: self.moveButton,

                    fixPosition: function(props) {
                        var
                        width = this.width,
                        height = this.height,
                        top = props.top,
                        left = props.left,
                        right = left + width,
                        bottom = top + height,
                        deltaLR = width < 300 ? width : 300,
                        deltaTB = height < 30 ? height: 30;

                        if (right > canvasWidth + deltaLR || props.left + width > canvasWidth + deltaLR) {
                            props.left = canvasWidth - width + deltaLR; // right dimension
                        }
                        if (left < - deltaLR || props.left < - deltaLR) {
                            props.left = - deltaLR; // left dimension
                        }

                        if (top > canvasHeight - 10 || props.top > canvasHeight - 10) {
                            props.top = canvasHeight - 10; // bottom dimension
                        }

                        if (props.top < 0 || top < 0) {
                            props.top = 0; // top dimension
                        }

                        if (data.align === "right") {
                            props.right = canvasWidth - this.width - props.left;
                            props.left = "auto";
                        } else {
                            props.right = "auto";
                        }

                        return props;
                    },

                    onMove: function(props) {
                        props = this.fixPosition(props);
                        $.extend(data, props);
                        module.redraw(self.el);
                        frame.css(props);

                        if (data.center) 
                            data.center = false;
                    },
                    onStart: function() {
                        module.hideSettingsMenu();
                        frame.addClass(CLASSES.INACTIVELAYER);

                        // store dimension of layer at the start of drag 
                        this.width = frame.width();
                        this.height = frame.height();

                        // convert the current layer to use left position instead of right
                        // to be compatible with dragger. We will convert back to using right when done.
                        if (data.align === "right") {
                            var right = parseInt(frame.css("right"), 10);
                            frame.css({
                                left: canvasWidth - this.width - right,
                                right: "auto"
                            });
                        }
                    },
                    onEnd: function(props) {}
                });
            },

            // Set z-index for a layer
            setIndex: function(index) {
                this.data["z-index"] = index;
                this.frame = this.frame || this.el;
                this.frame.css("z-index", index);
                this.index = index;
                return this;
            },

            // Remove layer. TODO: destroy all nodes, references, bindings, etc.
            remove: function() {
                this.frame.remove();
                this.parent.removeLayer(this);
            },

            saveData: function() {},

            getData: function() {
                return this.data;
            }
        });



        /**
         * Text Layer Controller, subclass of Layer Controller
         */
        LayerController['text'] = LayerController['base'].create({

            /**
            	 * Layer template (pulled from view.hsb)
            	 */
            tmpl: Handlebars.compile( '<div id="{{id}}" class="w-header-layer w-header-layer-text w-header-layer-align-{{align}} {{#if center}}w-header-layer-centered{{/if}}" style="{{#unless center}}left:{{left}}px; right:{{right}}px; {{/unless}} top:{{top}}px; z-index: {{z-index}};"> <div class="w-header-layer-container"> <div class="w-header-text-container"> <div class="w-header-text {{#if font}}w-font-{{font}}{{/if}}" data-placeholder="{{#l}}webs.module.header_editor.placeholder.text{{/l}}"> {{#if text}} {{{text}}} {{else}} {{#l}}webs.module.header_editor.placeholder.text{{/l}} {{/if}}</div> </div> </div> </div>'),


            /**
            	 * DOM event delegations
            	 */
            _events: [
            ["mousedown", ".w-header-text", "textContainerDown"],
            ["mouseenter", ".w-header-text", "textContainerEnter"],
            ["mouseleave", ".w-header-text", "textContainerLeave"]
            ],
            textContainerDown: function() {
                this.activate();
            },

            textContainerEnter: function() {
                this.parent.detachPanner();
            },

            textContainerLeave: function() {
                this.parent.attachPannerSafe();
            },


            /**
            	 * Initializer, called from initBase()
            	 */
            onInitBase: function() {
                // null text means placeholder
                if (typeof this.data.text !== 'string') 
                    this.data.text = PLACEHOLDERTEXT[this.type];

                if (this.onInitText) 
                    this.onInitText();
            },

            onRender: function() {
                var self = this;
                var type = this.type;
                var data = this.data;

                this.contentEl = this.el.find(".w-header-" + type).data('placeholder', PLACEHOLDERTEXT[type]);
                this.contentEl.attr('contenteditable', true);
                this.frame.addClass("w-header-layer-editing");

                var html = this.data.text === this.el.data('placeholder') ? null : this.data.text;
                this.controls = bldr.controls.create('text', {
                    el: this.contentEl,
                    html: html,
                    toolbarName: 'header_text',
                    linksData: data.linksData,
                    wizzy: {
                        plainText: true,
                        enableAlignment: false // disables wizzy-ui updating alignment button from style attr
                    }
                });
                this.controls.toolbar.bind({
                    'SetFont': function(val) {
                        self.setFont(val);
                    },
                    'Alignment': function(val) {
                        self.setAlignment(val);
                    },
                    'RemoveFormat': function() {
                        // When removing format, check to see if the whole container is selected
                        // If so, we also remove the typeface
                        var currRange = self.controls.wizzy.getRange();
                        var fullRange = currRange.cloneRange();
                        fullRange.selectNodeContents(self.controls.wizzy.el);
                        if (fullRange.equals(currRange)) {
                            self.setFont("__default__");
                            self.controls.toolbar.set("SetFont", "__default__");
                        }
                        self.controls.wizzy.exec("RemoveFormat");
                    }
                });

                // Themes can set a title to be default centered which places the text in the middle of the
                // page. The minute a user edits the header editor, we reset this value while keeping the text centered
                if (this.data.center) {
                    self.setAlignment("center");
                    this.data.center = false;
                    this.frame.removeClass("w-header-layer-centered");
                    this.frame.css("left", this.data.left = this.frame.position().left - this.frame.width() / 2);
                }
            },

            setAlignment: function(val) {
                var
                css = {}, prop = {},
                width = this.el.width(),
                canvasWidth = this.parent.dimension.width,
                position = this.el.position();

                prop.align = val;
                if (val === "left") {
                    prop.right = css.right = "auto";
                    prop.left = css.left = 0;
                } else if (val === "center") {
                    css.left = prop.left = canvasWidth / 2;
                    css.right = prop.right = "auto";
                } else {
                    prop.left = css.left = "auto";
                    prop.right = css.right = 0;
                }

                if (this.data.align) 
                    this.el.removeClass("w-header-layer-align-" + this.data.align);
                $.extend(this.data, prop);
                this.el.addClass("w-header-layer-align-" + val).css(css);
                return false;
            },

            // toolbar binding
            setFont: function(val) {
                var textEl = this.contentEl.attr('class', 'w-header-text w-header-' + this.type);

                if (this.data.font) 
                    textEl.removeClass('w-font-' + this.data.font);
                if (val) 
                    textEl.addClass('w-font-' + val);
                this.data.font = val;
            },

            /**
            	 * SITEBUILDER-1750 (and FWB-34782) for center-aligned text layers, when editing text, we have to
            	 * shift the text container by 50% of its width. This creates the effect of center
            	 * text position as it grows on both sides. When we "hide" text controls, we need to
            	 * convert this back to left positioning so the dead space created by the 50% offset
            	 * doesn't cover other layers
            	 */
            onActivate: function() {
                this.showTextControls();

                var width = this.frame.width();
                var css = this.frame.position();

                if (this.data.align === "center") {
                    css.right = "auto";
                    css.left += width / 2;
                    this.frame.css(css);
                }
            },
            onDeactivate: function() {
                //this.frame.removeClass("w-header-layer-editing");
                this.hideTextControls();

                var width = this.frame.width();
                var css = this.frame.position();

                if (this.data.align === "center") {
                    css.right = "auto";
                    css.left -= width / 2;
                    this.frame.css(css);
                    $.extend(this.data, css);
                    this.parent.dirty();
                }

                this.contentEl.attr('contenteditable', true);
                this.saveData();
            },

            showTextControls: function() {
                if (this.editing) 
                    return false;
                this.editing = true;

                // shows the text edit controls
                this.controls.show({
                    "SetFont": this.data.font || "__default__",
                    "Alignment": this.data.align
                });
            },

            hideTextControls: function() {
                if (!this.editing) 
                    return false;
                this.editing = false;

                this.frame.dragger('show');

                // restore with default text if empty
                if ($.trim(this.contentEl.text()) === "") {
                    this.contentEl.html(PLACEHOLDERTEXT[this.type]);
                }
            },

            onDestroy: function() {
                try {
                    if (this.controls && typeof this.controls.destroy === "function") 
                        this.controls.destroy();
                } catch (err) {}
            },

            saveData: function() {
                this.data.text = this.getData();
            },

            getData: function() {
                return this.controls.getData().html;
            }
        });



        /**
         * Heading Layer Controller, subclass of Text Layer Controller
         */
        LayerController['heading'] = LayerController['text'].create({
            onInitText: function() {
                var index = this.data["z-index"];
                var display = this.data["display"];
                this.setIndex(index).setDisplay(display);
                this.updateButton();
            },
            setDisplay: function(val) {
                this.frame.css("display", val);
                return this;
            },
            updateButton: function() {
                if (this.data.display === "none") 
                    this.button.removeClass("active");
                else 
                    this.button.addClass("active");
            },
            toggle: function() {
                this.data.display = this.data.display === "none" ? "block" : "none";
                this.setDisplay(this.data.display);
                this.updateButton();
            },
            remove: function() {
                this.toggle();
            }
        });
        LayerController['title'] = LayerController['heading'].create({
            tmpl: Handlebars.compile(' <div id="{{id}}" class="w-header-layer w-header-layer-text w-header-layer-align-{{align}} {{#if center}}w-header-layer-centered{{/if}}" style="{{#unless center}}left:{{left}}px; right:{{right}}px; {{/unless}} top:{{top}}px; z-index: {{z-index}}; display: {{display}}"> <div class="w-header-layer-container"> <h1 class="w-header-text-container"> <div class="w-header-text w-header-title {{#if font}}w-font-{{font}}{{/if}}" data-placeholder="{{#l}}webs.module.header_editor.placeholder.title{{/l}}"> {{#if text}} {{{text}}} {{else}} {{#l}}webs.module.header_editor.placeholder.title{{/l}} {{/if}} </div> </h1> </div> </div>'),
            saveData: function() {
                var newData = this.getData() || "";
                if (this.data.text !== newData) {
                    Spine.Events.trigger("headereditor:title:update", newData);
                }
                this.data.text = newData;
            }
        });
        LayerController['subtitle'] = LayerController['heading'].create({
            tmpl: Handlebars.compile('<div id="{{id}}" class="w-header-layer w-header-layer-text w-header-layer-align-{{align}} {{#if center}}w-header-layer-centered{{/if}}" style="{{#unless center}}left:{{left}}px; right:{{right}}px; {{/unless}} top:{{top}}px; z-index: {{z-index}}; display: {{display}}"> <div class="w-header-layer-container"> <h2 class="w-header-text-container"> <div class="w-header-text w-header-subtitle {{#if font}}w-font-{{font}}{{/if}}" data-placeholder="{{#l}}webs.module.header_editor.placeholder.subtitle{{/l}}"> {{#if text}} {{{text}}} {{else}} {{#l}}webs.module.header_editor.placeholder.subtitle{{/l}} {{/if}} </div> </h2> </div> </div>'),
            saveData: function() {
                var newData = this.getData() || "";
                if (this.data.text !== newData) {
                    Spine.Events.trigger("headereditor:subtitle:update", newData);
                }
                this.data.text = newData;
            }
        });



        /**
         * Image Controller, subclass of Base Layer Controller
         */
        LayerController['image'] = LayerController['base'].create({

            /**
            	 * Layer template (pulled from view.hbs)
            	 */
            tmpl: Handlebars.compile(
            ' {{#wrapInFrame}} ' +
            '<div id="{{id}}" class="w-header-layer w-header-layer-image" style="left:{{left}}px; top:{{top}}px; z-index: {{z-index}}; width:{{width}}px; height:{{height}}px;"> ' +
            '<div class="w-header-layer-container {{#unless url}}w-header-image-placeholder{{/unless}}">' +
            '<div class="w-header-tooltip" data-tooltip="{{#l}}webs.module.header_editor.placeholder.image{{/l}}" >' +
            '{{#if url}}' +
            '<img{{#if alt}} alt="{{alt}}"{{/if}} class="w-header-image" src="{{manipulate_image . fitWidth="1920" fitHeight="1920"}}"/> ' +
            '{{else}} ' +
            '<div class="placeholder">' +
            '<div class="w-header-placeholder-icon"></div>' +
            '<p>{{#l}}webs.module.header_editor.placeholder.image{{/l}}</p>' +
            '</div>' +
            '<img class="w-header-image"/>' +
            '{{/if}} ' +
            '</div>' +
            '</div> ' +
            '</div> ' +
            '{{/wrapInFrame}}'),

            /**
            	 * DOM event delegations
            	 */
            _events: [
            ["mousedown", "", "imageDown"],
            ["dblclick", ".w-header-layer-container", "imageDblclick"]
            ],
            layerUp: function() {
                this.parent.changeLayerZ(this, 1);
                return false;
            },
            layerDown: function() {
                this.parent.changeLayerZ(this, - 1);
                return false;
            },
            imageDblclick: function() {
                hideImageLayerTooltip();
                this.toolbar.click('image');
            },
            imageDown: function() {
                var module = this.parent;
                if (!module.activeLayer || this !== module.activeLayer) {
                    this.activate();
                    return false;
                }
            },


            /**
            	 * Initializer, called from initBase()
            	 */
            onInitBase: function() {
                var self = this;

                this.data.position = "absolute";

                this.toolbar = bldr.toolbar.create('header_layer_image');

                this.toolbar.bind({
                    'image': function(obj) {
                        self.changeImageHandler(obj);
                    },
                    'imageLink': function(obj) {
                        self.changeLinkHandler(obj);
                    },
                    'removeLink': function() {
                        self.changeLinkHandler(null);
                    },
                    'setFrame': function(slug) {
                        self.setFrame(slug);
                    },
                    'layerUp': function() {
                        self.layerUp();
                    },
                    'layerDown': function() {
                        self.layerDown();
                    }
                });

                this.linkGroup = this.toolbar.find(".w-header_editor-linkgroup");
                this.linkButton = this.linkGroup.find('.w-header_editor-link');

            },

            onRender: function() {
                var data = this.data;
                this.contentEl = this.el.find(".w-header-image");
                this.toolbar.setFeedback(this.container);
                this.attachResizer();
            },



            /**
            	 * Toolbar event bindings
            	 */
            updateToolbarLink: function() {
                this.linkButton.removeClass("active");
                this.linkGroup[this.data.linkInfo ? "removeClass" : "addClass"]("canceled");
            },
            changeLinkHandler: function(obj) {
                this.data.linkInfo = obj;
                this.updateToolbarLink();
            },
            changeImageHandler: function(obj) {
                var self = this;

                if (obj.url) {
                    this.killPlaceholderClass();
                    this.el.find('.placeholder').hide();
                    this.el.find('.w-header-image').show();
                    self.contentEl.attr({
                        "src": "/static/projects/finch/images/header_image_loading.gif"
                    });
                    self.el.css({
                        "width": 74,
                        "height": 74
                    });
                    var $img = $('<img/>');
                    $img.load(function() {
                        setTimeout(function() {
                            var img = $img[0];
                            self.updateImage({
                                data: obj,
                                width: img.width || 100,
                                height: img.height || 100
                            });
                        }, 1);
                    });
                    $img.attr('src', obj.url);
                }
            },

            killPlaceholderClass: function() {
                this.el.find('.w-header-layer-container').removeClass('w-header-image-placeholder');
            },

            setFrame: function(slug) {
                if (this.data.frame === slug || (!this.data.frame && slug === true)) 
                    return false;

                if (typeof slug !== 'string' && typeof this.data.frame !== "undefined") 
                    delete this.data.frame;
                else 
                    this.data.frame = slug;

                this.render();
            },

            attachResizer: function() {
                var self = this;
                var startData, moveData;

                var data = this.data;
                var layer = this.el;
                var frame = this.frame;
                var module = this.parent;

                layer.resizer({
                    maxWidth: data.fullWidth,
                    minWidth: Math.max(data.width / 20, 16),
                    minHeight: Math.max(data.height / 20, 16),
                    onStart: function() {
                        var pos = frame.position();
                        startData = {
                            width: frame.width(),
                            height: frame.height(),
                            left: parseInt(pos.left, 10),
                            top: parseInt(pos.top, 10)
                        };
                        frame.addClass(CLASSES.INACTIVELAYER);
                    },
                    onMove: function(corner, newWidth, newHeight) {
                        var newLeft, newTop, ratio = newWidth / newHeight;

                        if (newHeight + startData.top > module.dimension.height) {
                            newHeight = module.dimension.height - moveData.top;
                            newWidth = newHeight * ratio;
                            layer.css({
                                width: newWidth,
                                height: newHeight 
                            });
                            frame.css({
                                left: moveData.left,
                                top: moveData.top 
                            });
                            return false;
                        } else {
                            if (corner === 'left') {
                                newLeft = startData.left - (newWidth - startData.width);
                                newTop = startData.top;
                            }
                            if (corner == 'right') {
                                newLeft = startData.left;
                                newTop = startData.top;
                            } else if (corner === 'center') {
                                // corner center doesn't exist, but if it did...
                                newLeft = startData.left - parseInt((newWidth - startData.width) / 2, 10);
                                newTop = startData.top - parseInt((newHeight - startData.height) / 2, 10);
                            }
                        }


                        frame.css({
                            left: newLeft,
                            top: newTop 
                        });
                        moveData = {
                            left: newLeft,
                            top: newTop,
                            width: newWidth,
                            height: newHeight
                        };
                    },
                    onEnd: function() {
                        $.extend(data, moveData);
                        frame.removeClass(CLASSES.INACTIVELAYER);
                    }
                });
            },

            // Callback when controls performs any kind of udpate to the image
            updateImage: function(newData) {
                var data = this.data, ratio = newData.width / newData.height;

                data.url = newData.data.url;
                data.imageType = newData.data.imageType;
                if (newData.data[data.imageType])
                    data[data.imageType] = newData.data[data.imageType];
                data.fullWidth = newData.width;
                data.fullHeight = newData.height;
                data.alt = undefined;

                if (data.width > newData.width) {
                    data.width = newData.width;
                    data.height = newData.height;
                } else {
                    data.width = parseInt(data.height * ratio, 10);
                }
                this.el.css({
                    'width': data.width,
                    'height': data.height 
                });
                this.frame.resizer('destroy');
                this.contentEl.attr('src', newData.data.url);
                this.attachResizer();
                this.attachDragger();
            },

            onActivate: function() {
                var toolbarData = {
                    "imageLink": this.data.linkInfo || {},
                    "frame": this.data.frame,
                    "image": {
                        url: this.data.url || "http://static.websimages.com/static/projects/finch/images/header_image_placeholder.png",
                        imageType: this.data.imageType
                    }
                };
                toolbarData.image[this.data.imageType] = this.data[this.data.imageType];
                this.toolbar.show(toolbarData);
                this.updateToolbarLink();
            },

            onDestroy: function() {
                try {
                    if (this.frame && this.frame.resizer) 
                        this.frame.resizer("destroy");
                } catch (err) {}
            }

        });

        LayerController['logo'] = LayerController['image'].create({

            /**
            	 * DOM event delegations
            	 */
            _events: [
            ["mousedown", "", "imageDown"],
            ["dblclick", ".w-header-layer-container", "imageDblclick"]
            ],
            /**
            	 * Layer template (shamefully copied from hbs template)
            	 */
            tmpl: Handlebars.compile(
            ' {{#wrapInFrame}} ' +
            '<div id="{{id}}" class="w-header-layer w-header-layer-image w-header-layer-logo" style="left:{{left}}px; top:{{top}}px; width:{{width}}px; height:{{height}}px; display:{{display}};"> ' +
            '<div class="w-header-layer-container {{#unless url}}w-header-logo-placeholder{{/unless}}"> ' +
            '<div class="w-header-tooltip" data-tooltip="{{#l}}webs.module.header_editor.placeholder.image{{/l}}" >' +
            '{{#if url}} ' +
            '<img class="w-header-image w-header-logo" src="{{manipulate_image . fitWidth="1920" fitHeight="1920"}}"/> ' +
            '{{else}} ' +
            '<div class="placeholder"> ' +
            '<div class="w-header-placeholder-icon"></div>' +
            '<p>' +
            '{{#l}}webs.module.header_editor.placeholder.image{{/l}}' +
            '</p> ' +
            '</div>' +
            '<img class="w-header-image w-header-logo"/>' +
            '{{/if}} ' +
            '</div>' +
            '</div>' +
            '</div>' +
            '{{/wrapInFrame}}'),

            setDisplay: function(val) {
                this.frame.css("display", val);
                return this;
            },
            updateButton: function() {
                if (this.data.display === "none") 
                    this.button.removeClass("active");
                else 
                    this.button.addClass("active");
            },
            toggle: function() {
                this.data.display = this.data.display === "none" ? "block" : "none";
                this.setDisplay(this.data.display);
                this.updateButton();
            },
            remove: function() {
                this.toggle();
            },

            onInitBase: function() {
                var self = this;

                this.data.position = "absolute";

                this.toolbar = bldr.toolbar.create('header_layer_logo');

                this.toolbar.bind({
                    'image': function(obj) {
                        self.changeImageHandler(obj);
                    },
                    'imageLink': function(obj) {
                        self.changeLinkHandler(obj);
                    },
                    'removeLink': function() {
                        self.changeLinkHandler(null);
                    },
                    'setFrame': function(slug) {
                        self.setFrame(slug);
                    }
                });

                this.linkGroup = this.toolbar.find(".w-header_editor-linkgroup");
                this.linkButton = this.linkGroup.find('.w-header_editor-link');
            },

            killPlaceholderClass: function() {
                this.el.find('.w-header-layer-container').removeClass('w-header-logo-placeholder');
            }
        });








        /**
         * This module uses an element pattern.
         * It holds an array (this.layerControllers) of controllers which describes a layer on the canvas
         * The array index is also the z-index value for the layer it holds
         */
        $.extend(module, {

            safeOneLoaded: function() {
                this.initTooltip();
            },

            initTooltip: function() {
                var self = this;
                try {
                    this.$tt = bldr.pageController.$.tooltip({
                        content: translate('webs.module.header_editor.tooltip')
                    });
                    this.el.bind("mouseover", function() {
                        if (!bldr.pageController.dom.$body.find(".webs-bin-active").length)
                            self.$tt.addClass("active");
                    });
                    this.el.bind("mouseout", function() {
                        self.$tt.removeClass("active");
                    });
                } catch (err) {
                    webs.log.error("toolip init error in header editor", err);
                }
            },

            toolbar: {},

            onActivate: function(event) {
                var self = this;
                var bldr = parent.bldr;

                if (this.$tt)
                    this.$tt.removeClass("active");

                // Force the z-index for title and subtitle
                this.data.title['z-index'] = 20;
                this.data.subtitle['z-index'] = 21;

                this.$ = bldr.pageController.$;
                this.$canvas = this.el.find('.w-header-area'); //canvas - parent node for all the layers

                this.activeLayer = null; // Controller reference to the current active layer
                this.$background = null; // jQuery reference to the header's background whose ID the template defines

                this.fixedButtons = {}; // Button corresponding to a fixed element
                this.fixedControllers = {}; // Key (type) to non-removable controllers {'title', 'subtitle'}
                this.layerControllers = []; // Object of z-index value as key and controller as value.

                // Store dimension of canvas
                this.dimension = {
                    width: this.$canvas.width(),
                    height: this.$canvas.height()
                };

                // Toolbar
                this.toolbar.header = bldr.toolbar.create('header_editor');
                this.toolbar.header.show();

                // Modulebar
                this.renderModulebar();

                setTimeout(function() {
                    self.updateBackground();
                }, 10);

                // Populate the layers
                this.addLayers();

                // Disable link in text layers
                this.el.delegate('.w-header-text a', 'click', function() {
                    return false;
                });

                // Reflow modulebar container for IE8
                if (bldr.IE <= 8) 
                    this.redraw(this.moduleBar.container);

                // If header edit area is too close to toolbar, nudge page down
                // 30px for the toolbar, 50px for the space needed to allow layer to be
                // dragged 30px above top of header and 20px space for layer title/drag bar
                var canvasOffsetTop = this.$canvas.offset().top;
                if (canvasOffsetTop < 80) {
                    this._paddedTop = true;
                    this.$html = bldr.pageController.dom.$html;
                    this.$html.animate({
                        "padding-top": 80 - canvasOffsetTop
                    }, 100);
                }
            },

            onDeactivate: function() {
                this.el.removeClass('w-header-gridon');
                if (this._paddedTop) 
                    this.$html.css("padding-top", "");
                if (this.$dimensionTT && this.$dimensionTT.jquery) 
                    this.$dimensionTT.remove();
                this.deactivateLayer();
                this.destroyAllLayers();
            },

            destroyAllLayers: function() {
                this.fixedControllers["title"].destroy();
                this.fixedControllers["subtitle"].destroy();
                this.fixedControllers["logo"].destroy();
                for (var i = 0, len = this.layerControllers.length; i < len; i++) {
                    if (this.layerControllers[i] && this.layerControllers[i].destroy) {
                        this.layerControllers[i].destroy();
                    }
                }
            },

            attachControls: function() {
                var self = this;
                var $canvas = this.$canvas;
                $canvas.unbind("mousedown").unbind("mouseup");
                $canvas.mousedown(function(e) {
                    if ($canvas.find(e.target)[0]) {
                        var ctrl = self.activeLayer;
                        // Clicked on an element inside $canvas
                        if (!(ctrl && ctrl.contains(e.target))) {
                            //clicked element inside canvas that isn't a member of the active layer
                            self.deactivateLayer();
                        }
                    } else {
                        self.deactivateLayer();
                    }
                }).mouseup(function() {
                    $canvas.find('.' + CLASSES['INACTIVELAYER']).removeClass(CLASSES['INACTIVELAYER']);
                });
            },

            renderModulebar: function() {
                var self = this;
                var data = this.data;


                var $background = this.$background = this.$('.webs-header-bg').data('data', this.data.background);
                var moduleBar = this.moduleBar = bldr.pageController.getModuleBar();
                var mbRight = moduleBar.buttonContainerRight;
                var mbContainer = this.moduleBar.container.addClass('bldr-modulebar-bottom');
                var mbWidth = mbContainer.width();

                // checks if the header's width is too small to fit toggles on modulebar
                // if background exists, the 'change background' and toggle switch takes up about 300 px
                var isSmall = mbWidth < 485 + ($background[0] ? 300 : 0);

                // shared buttons in both small and normal width headers
                var addTextObj = {
                    label: 'Add Text',
                    name: 'addText',
                    btnPrefix: "w-btn4",
                    iconElement: true,
                    classNames: "w-header_editor-text"
                };
                var addImageObj = {
                    label: 'Add Image',
                    name: 'addImage',
                    btnPrefix: "w-btn4",
                    iconElement: true,
                    classNames: "w-header_editor-image"
                };
                var changeBgObj = {
                    label: 'Change Background',
                    name: 'changeBackground',
                    style: 'blue',
                    dialog: {
                        url: "/s/sitebuilder/dialogs/image",
                        localizeHeading: "webs.bldr.toolbar.text.ChangeImage",
                        width: 650,
                        height: 361
                    }
                };
                if (isSmall) {
                    //changeBgObj.icon = 'headerbtn_bg';
                    changeBgObj.label = 'Background';
                }
                this.fixedButtons = {
                    grid: this.moduleBar.addButton({
                        iconSet: 'w-btnstates',
                        icon: 'grid',
                        name: 'grid',
                        title: 'Toggle Grid',
                        label: 'Header Grid' 
                    })
                };
                mbRight.append( this.fixedButtons.grid);


                // BG reference, only show background button and toggle if background container exists
                if ($background[0]) {
                    moduleBar.addButton({
                        type: 'divider' 
                    });
                    var bgButtonContainer = moduleBar.addButton(changeBgObj), dimTT;
                    var bgButton = bgButtonContainer.children().attr("title", ""); // removing title attr here bc it covers the tooltip
                    try {
                        dimTT = bldr.pageController.$.tooltip({
                            content: translate("webs.module.header_editor.dimensions", {
                                width: this.$canvas.width(),
                                height: this.$canvas.height()
                            }) 
                        });
                        bgButton.bind("mouseenter", function() {
                            dimTT.addClass("active");
                        });
                        bgButton.bind("mouseleave", function() {
                            dimTT.removeClass("active");
                        });
                        this.$dimensionTT = dimTT.css("z-index", 130); // higher z-index to overlap the modulebar
                    } catch (err) {
                        webs.log.error("toolip init error in header editor", err);
                    }
                }




                // toggle switches
                var titleObj = {
                    type: 'switch',
                    label: 'Title',
                    name: 'toggleTitle',
                    classNames: 'w-switch-title ' + (data.title.display === "block" ? 'active' : '')
                };
                var subtitleObj = {
                    type: 'switch',
                    label: 'Tagline',
                    name: 'toggleSubtitle',
                    classNames: 'w-switch-subtitle ' + (data.subtitle.display === "block" ? 'active' : '')
                };
                var logoObj = {
                    type: 'switch',
                    label: "Logo",
                    name: 'toggleLogo',
                    classNames: 'w-switch-logo ' + (data.logo.display === "block" ? 'active' : '')
                };
                var bgObj = {
                    type: 'switch',
                    label: 'Background',
                    name: 'toggleBackground',
                    classNames: 'w-switch-bg ' + (data.showBackground ? 'active' : '')
                };

                if (!isSmall) {
                    this.fixedButtons.logo = this.moduleBar.addButton(logoObj);
                    this.fixedButtons.title = this.moduleBar.addButton(titleObj);
                    this.fixedButtons.subtitle = this.moduleBar.addButton(subtitleObj);

                    if ($background[0]) {
                        this.fixedButtons.background = this.moduleBar.addButton(bgObj);
                        mbRight.append(this.fixedButtons.background);
                        this.updateBackground();
                    }
                    moduleBar.addButton({
                        type: 'divider' 
                    });
                    var moreContainer = this.moduleBar.addButton({
                        type: 'button-grid',
                        labelIntl: 'webs.bldr.modules.header_editor.toolbar.addContent',
                        menuClassNames: ['w-header_editor-addContent'],
                        classNames: ["w-header_editor-addContent-button"],
                        style: 'blue',
                        menuPosition: ['top', 'center'],
                        items: [
                        addTextObj,
                        addImageObj
                        ]
                    });
                    instances = moreContainer.find('.w-more_container').data('toolbar_instances');
                    this.fixedButtons.image = instances.addImage;
                    this.fixedButtons.text = instances.addText;
                } else {
                    mbContainer.addClass("bldr-modulebar-header-small");

                    var headerMoreMenu = [titleObj, subtitleObj, logoObj];
                    if ($background[0]) 
                        headerMoreMenu.push(bgObj);

                    var switchDef = {
                        type: 'switch-bare',
                        display: 'inline' 
                    };
                    $.extend(titleObj, switchDef);
                    $.extend(subtitleObj, switchDef);
                    $.extend(logoObj, switchDef);
                    $.extend(bgObj, switchDef);

                    moduleBar.build({
                        module: self,
                        width: 172,
                        title: "Header Settings",
                        menu: headerMoreMenu
                    });
                    this.fixedButtons.text = this.moduleBar.addButton({
                        label: '+ Text',
                        name: 'addText',
                        style: 'blue',
                        icon: 'headerbtn_text' 
                    }).children();
                    this.fixedButtons.image = this.moduleBar.addButton({
                        label: '+ Image',
                        name: 'addImage',
                        style: 'blue',
                        icon: 'headerbtn_image'
                    }).children();
                    this.fixedButtons.title = mbContainer.find(".w-switch-title");
                    this.fixedButtons.subtitle = mbContainer.find(".w-switch-subtitle");
                    this.fixedButtons.logo = mbContainer.find(".w-switch-logo");
                    if ($background[0]) 
                        this.fixedButtons.background = mbContainer.find(".w-switch-bg");
                    mbRight.append( this.fixedButtons.grid);
                }





                var canvasWidth = self.el.width();
                var canvasHeight = self.el.height();
                var placeholderHeight = Math.min(74, canvasHeight);
                var placeholderWidth = placeholderHeight;
                var modulebarBindings = {
                    grid: function() {
                        if (self.el.hasClass(CLASSES.GRIDON)) {
                            self.el.removeClass(CLASSES.GRIDON);
                            self.fixedButtons.grid.children().removeClass('active');
                        } else {
                            self.el.addClass(CLASSES.GRIDON);
                            self.fixedButtons.grid.children().addClass('active');
                        }
                    },
                    toggleSubtitle: function(on) {
                        self.toggleFixedLayer('subtitle');
                        Spine.Events.trigger("headereditor:subtitle:toggle", on);
                    },
                    toggleTitle: function(on) {
                        self.toggleFixedLayer('title');
                    },
                    toggleLogo: function(on) {
                        self.toggleFixedLayer('logo');
                    },
                    toggleBackground: function(on) {
                        self.data.showBackground = on;
                        self.updateBackground();
                    },

                    addLayer: function(stack, type, data) {
                        if (stack.length > 9) 
                            return false;
                        data['z-index'] = stack.length;
                        stack.push(data);
                        if (stack.length === 10) 
                            self.fixedButtons[type].parent().addClass(CLASSES.BUTTONDISABLED);

                        self.addLayer({
                            type: type,
                            stack: stack,
                            data: data 
                        });
                        self.unshuffleZ();
                    },

                    addText: function() {
                        modulebarBindings.addLayer(self.data.text_layers, 'text', {
                            "text": null,
                            "align": "left",
                            "center": false,
                            "top": canvasHeight / 2,
                            "left": canvasWidth / 2 - placeholderWidth,
                            "right": "auto",
                            "z-index": ZRANGE.text - 1
                        });
                    },
                    addImage: function() {
                        modulebarBindings.addLayer(self.data.image_layers, 'image', {
                            "url": null,
                            "width": placeholderWidth,
                            "height": placeholderHeight,
                            "fullWidth": 300,
                            "fullHeight": 180,
                            "top": canvasHeight / 2 - placeholderHeight / 2,
                            "left": canvasWidth / 2 - placeholderWidth / 2,
                            "z-index": ZRANGE.image - 1
                        });
                    },
                    changeBackground: function(obj) {
                        if ($background.length) {
                            self.data.background.url = obj.url;
                            self.data.showBackground = true;

                            self.fixedButtons.background.children().addClass('active');
                            self.updateBackground();
                        }
                    }
                };
                this.moduleBar.addListener(modulebarBindings);
                this.moduleBar.setFeedback($background);
            },


            hideSettingsMenu: function() {
                if (!this.settingsMenu) 
                    this.settingsMenu = this.moduleBar.buttonContainerRight.find(".w-more");
                if (this.settingsMenu.length) 
                    this.settingsMenu.removeClass("active");
            },



            /**
            	 * Toggles a layer that cannot be removed
            	 * @param	type     String specifying the type of layer. Right now either title or subtitle
            	 */
            toggleFixedLayer: function(type) {
                if (this.fixedControllers[type]) {
                    this.fixedControllers[type].toggle();
                    this.redraw(this.el);
                    return true;
                }
                return false;
            },


            /**
            	 * Deletes a layer from data and $canvas
            	 * @param	ctrl	Controller reference to the layer to remove
            	 */
            removeLayer: function(ctrl) {
                var
                index = ctrl.index,
                stack = ctrl.stack,
                type = ctrl.type,
                lastData = stack[stack.length - 1],
                lastCtrl = this.layerControllers[stack.length - 1];

                // Only remove from data if layer isn't "fixed" like title/subtitle
                if (!this.toggleFixedLayer(type)) {
                    // If not the last element, move last element to its position
                    var indexInStack = stack.indexOf(ctrl.data);
                    if (indexInStack !== stack.length - 1) {
                        stack[indexInStack] = lastData;
                        this.layerControllers[index] = lastCtrl;
                    }

                    // Pop last element out of stack
                    stack.pop();
                    this.layerControllers.pop();

                    // enable add layer button if stack is no longer full
                    this.checkPreventAddLayer();
                }
            },


            /**
            	 * Grays out the "Add (text|image)" button when the stack is full
            	 */
            checkPreventAddLayer: function(type) {
                var textButton = this.fixedButtons['text'].parent();
                var imageButton = this.fixedButtons['image'].parent();

                if (this.data['text_layers'].length >= 9) 
                    textButton.removeClass(CLASSES.BUTTONDISABLED);
                if (this.data['image_layers'].length >= 9) 
                    imageButton.removeClass(CLASSES.BUTTONDISABLED);
            },



            /**
            	 * Changes the z-index overlapping of a layer
            	 * @param	ctrl		Controller reference to the layer to move
            	 * @param	direction	Negative/positive integer represents direction of the move
            	 */
            changeLayerZ: function(ctrl, direction) {
                var type = ctrl.type,
                data = ctrl.data,
                z = ctrl.index,
                r = ZRANGE[type],
                newZ = direction < 0 ? z - 1 : z + 1;
                old = this.layerControllers[newZ];

                if (typeof direction !== 'number' || direction === 0) 
                    return false;
                if ((direction < 0 && z === r[0]) || (direction > 0 && z === r[1] - 1)) 
                    return false;

                this.layerControllers[newZ] = ctrl.setIndex(newZ);
                data['z-index'] = newZ;
                this.layerControllers[z] = old;
                if (old) 
                    old.setIndex(z);
            },


            /**
            	 * Sets the z-index of a layer. If a layer with z-index exists, use next available index
            	 * Image layer takes 0-9, Text layer takes 10-19. Note: there are only 10 elements of each type allowed
            	 * @param	layer	Controller reference to the layer to set z-index on
            	 * @param	z		Integer z-index value
            	 * @return	boolean	true if there was available slot and operation was sucessful, false otherwise
            	 */
            setLayerIndex: function(ctrl, index) {
                var type = ctrl.type;
                var r = ZRANGE[type];
                var ctrls = this.layerControllers;

                if (r[0] <= index && index < r[1] && !ctrls[index]) {
                    ctrls[index] = ctrl;
                    ctrl.setIndex(index);
                    return true;
                } else {
                    // either z-index is out of range of z-index already exist	
                    for (var i = r[0]; i < r[1]; i++) {
                        if (!ctrls[i]) {
                            ctrls[i] = ctrl;
                            ctrl.setIndex(i);
                            return true;
                        }
                    }
                    return false;
                }
            },


            /**
            	 * Pushes layer z-index down so there's no gap in between z-indexes other than the gap from type range
            	 */
            unshuffleZ: function() {
                var k, r, i, j;
                var ctrls = this.layerControllers;

                // search all ranges
                for (k in ZRANGE) {
                    if (ZRANGE.hasOwnProperty(k)) {
                        r = ZRANGE[k];
                        // search all items in range
                        for (i = r[0]; i < r[1]; i++) {
                            // found a gap
                            if (!ctrls[i]) {
                                // search for next available layer
                                for (j = i + 1; j < r[1]; j++) {
                                    if (ctrls[j]) {
                                        ctrls[i] = ctrls[j];
                                        ctrls[i].setIndex(i);
                                        delete ctrls[j];
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            },

            // makes call to element/layer
            deactivateLayer: function(fromActivate) {
                if (this.activeLayer) 
                    this.activeLayer.deactivate(fromActivate);
                this.activeLayer = null;

                if (!fromActivate) 
                    this.attachPanner();
            },

            // called from element/layer
            activateLayer: function(ctrl) {
                this.activeLayer = ctrl;

                this.detachPanner();
            },

            addLayers: function() {
                var self = this;
                var data = this.data;
                var fixedButtons = this.fixedButtons;

                //clear current view before adding layer
                this.$canvas.empty();
                $.each(data['image_layers'], function(i, layer) {
                    self.addLayer({
                        type: 'image',
                        stack: data['image_layers'],
                        data: layer
                    });
                });
                $.each(data.text_layers, function(i, layer) {
                    self.addLayer({
                        type: 'text',
                        stack: data['text_layers'],
                        data: layer
                    });
                });
                this.addLayer({
                    type: 'title',
                    data: data.title 
                });
                this.addLayer({
                    type: 'subtitle',
                    data: data.subtitle 
                });
                this.addLayer({
                    type: 'logo',
                    data: data.logo 
                });
                if (data['image_layers'].length === 10) 
                    fixedButtons['image'].addClass(CLASSES['BUTTONDISABLED']);
                if (data['text_layers'].length === 10) 
                    fixedButtons['text'].addClass(CLASSES['BUTTONDISABLED']);
            },

            /**
            	 * Adds a layer to data and $canvas
            	 */
            addLayer: function(info) {

                var
                type = info.type,
                stack = info.stack,
                data = info.data,
                index = data['z-index'],
                ctrl = LayerController[type].init({
                    parent: this,
                    $: this.$,
                    type: type,
                    stack: stack,
                    data: data,
                    el: this.$("<div/>").appendTo(this.$canvas)
                });


                if (type === 'title' || type === 'subtitle' || type === 'logo') {
                    // title/subtitle, add controller to fixedControllers
                    this.fixedControllers[type] = ctrl;
                } else {
                    // If not title or subtitle layer, set & fix z-index
                    this.setLayerIndex(ctrl, index);
                }

                return ctrl;
            },


            detachPanner: function() {
                if (this.$canvas.panner) 
                    this.$canvas.panner("destroy");
            },
            attachPannerSafe: function() {
                if (!this.activateLayer) 
                    this.attachPanner();
            },
            attachPanner: function() {
                var $background = this.$background || this.$('.webs-header-bg'),
                dat = this.data.background, pos = dat.position, left, top;

                if ($background[0]) {
                    if (typeof pos !== "object" || !pos.hasOwnProperty("left") || !pos.hasOwnProperty("top")) {
                        dat.position = {
                            left: 0,
                            top: 0
                        };
                        pos = dat.position;
                    }
                    this.$canvas.panner({
                        reference: this.$canvas,
                        onStart: function(x, y) {
                            left = pos.left;
                            top = pos.top;
                        },
                        onMove: function(y, x, dy, dx) {
                            var newLeft = left + dx, newTop = top + dy;
                            dat.position = {
                                left: newLeft,
                                top: newTop
                            };
                            if ($background[0]) 
                                $background.css('background-position', newLeft + 'px ' + newTop + 'px');
                        }
                    });
                }
            },

            updateBackground: function() {
                if (!this.$background[0]) 
                    return false;

                var $background = this.$background || this.$('.webs-header-bg'),
                bgdata = this.data.background, pos = bgdata.position;

                if (this.data.showBackground && bgdata.url && $background.length) {
                    $background.css('background', 'url("' + bgdata.url + '") ' + pos.left + 'px ' + pos.top + 'px');
                    this.attachPanner();
                } else {
                    $background.css('background', 'none');
                    this.detachPanner();
                }

                if (bldr.IE <= 8) 
                    this.fixedButtons.background.hide().show(); // Force reflow in IE8
            },

            afterUndo: function() {
                this.updateBackground();
                if (this._isActivated) {
                    this.moduleBar.unbind();
                    this.moduleBar.buttonContainer.empty();
                    this.moduleBar.buttonContainerRight.children(':not(.bldr-modulebar-done-container)').remove();
                    this.activate();
                }
            },

            getData: function() {
                if (this.activeLayer) 
                    this.activeLayer.saveData();
                return this.data;
            }
        });


    })(mod, ext);

    return ModuleClassLoader.register('header_editor', mod, ext);
});




// fb-like Module (widget)

// breaks if require is used
define('module/fb-like', ['internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.fb-like'], function(ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function (module, extend) {

        // Show Cover?
        extend.shouldShowCover = function () {
            return false;
        };

        // Widget
        extend.isWidget = true;



        // Module HTML Template
        extend.template = '{{#if url}}\
        <iframe src="//www.facebook.com/plugins/likebox.php?href={{formatFacebookURL url}}&amp;width={{_width}}&amp;colorscheme={{colorScheme}}&amp;show_faces={{showFaces}}&amp;stream={{showStream}}&amp;header={{showHeader}}&amp;height={{#if showFaces}}{{#if showStream}}{{#if showHeader}}590{{else}}558{{/if}}{{else}}{{#if showHeader}}290{{else}}258{{/if}}{{/if}}{{else}}{{#if showStream}}{{#if showHeader}}427{{else}}395{{/if}}{{else}}62{{/if}}{{/if}}" scrolling="no" frameborder="0" style="border:none; overflow:hidden; width:{{_width}}px; height:{{#if showFaces}}{{#if showStream}}{{#if showHeader}}590px{{else}}558px{{/if}}{{else}}{{#if showHeader}}290px{{else}}258px{{/if}}{{/if}}{{else}}{{#if showStream}}{{#if showHeader}}427px{{else}}395px{{/if}}{{else}}62px{{/if}}{{/if}};" allowTransparency="true"></iframe>\
        {{else}}\
        <div class="bldr-placeholder-wrap" style="width:{{_width}}px; height:{{#if showFaces}}{{#if showStream}}{{#if showHeader}}590px{{else}}558px{{/if}}{{else}}{{#if showHeader}}290px{{else}}258px{{/if}}{{/if}}{{else}}{{#if showStream}}{{#if showHeader}}427px{{else}}395px{{/if}}{{else}}62px{{/if}}{{/if}};">\
        <div class="bldr-placeholder-like-box bldr-placeholder-social bldr-placeholder-element" data-placeholder-text="{{#l}}webs.module.fb-like.clicktoedit{{/l}}">\
        </div>\
        </div>\
        {{/if}}\
        ';

        // Module Default Data
        extend.defaultData = {
            "url": "",
            "showFaces": true,
            "showHeader": false,
            "showStream": false,
            "colorScheme": "light"
        };

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        module.getMinWidth = function() {
            return 220;
        };

        // Register Toolbars
        bldr.toolbar.register('fb-like', {
            "icon": "fb-like",
            "items": [{
                "type": "textfield",
                "name": "url",
                "label": "Facebook URL",
                "placeholder": "placeholderUrl",
                "width": "200"
            }, {
                "type": "menu",
                "label": "Settings",
                "menuLabel": "Advanced Settings",
                "menuClassNames": "w-fb-like-advanced",
                "menu": [{
                    "display": "inline",
                    "label": "Show Header",
                    "type": "switch-bare",
                    "name": "showHeader"
                }, {
                    "display": "inline",
                    "label": "Show Faces",
                    "type": "switch-bare",
                    "name": "showFaces"
                }, {
                    "display": "inline",
                    "label": "Show Stream",
                    "type": "switch-bare",
                    "name": "showStream"
                }, {
                    "type": "dropdown",
                    "label": "Color Scheme",
                    "name": "colorScheme",
                    "display": "inline",
                    "width": 80,
                    "options": [{
                        "label": "Light",
                        "value": "light"
                    }, {
                        "label": "Dark",
                        "value": "dark"
                    }
                    ]
                }
                ]
            }
            ]
        });

        // View JS


        // Edit JS
        module.safeOneLoaded = function() {
            this.chooseDarkOrLightStyle('dark', 'light');
            this.helptip = {
                "fieldName": "url",
                "message": translate("webs.module.fb-like.helptip")
            };
        };


    })(mod, ext);

    return ModuleClassLoader.register('fb-like', mod, ext);

});


// paypal Module (widget)

// breaks if require is used
define('module/paypal', ['internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.paypal'], function(ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function (module, extend) {

        // Show Cover?
        extend.shouldShowCover = function () {
            return false;
        };

        // Widget
        extend.isWidget = true;

        extend.popoverConfig = {
            "title": "webs.module.paypal.title",
            "header": "webs.module.paypal.header",
            "button_text": "webs.module.paypal.buttontext",
            "preview_width": 350,
            "field_defs": [{
                "name": "paypal_username",
                "type": "text",
                "label": "webs.module.paypal.field.username.label",
                "hint": "webs.module.paypal.field.username.hint",
                "validation": {
                    "pattern": "^\\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4})\\s*$",
                    "message": "webs.module.paypal.field.username.validationmsg"
                }
            }, {
                "name": "item_name",
                "type": "text",
                "label": "webs.module.paypal.field.item.label",
                "hint": "webs.module.paypal.field.item.hint",
                "validation": {
                    "pattern": "[^\\s]",
                    "message": "webs.module.paypal.field.item.validationmsg"
                }
            }, {
                "name": "price",
                "type": "currency",
                "label": "webs.module.paypal.field.price.label",
                "hint": "webs.module.paypal.field.price.hint",
                "accepted_currency": [{
                    "label": "USD",
                    "value": "USD"
                }, {
                    "label": "EUR",
                    "value": "EUR"
                }, {
                    "label": "CAD",
                    "value": "CAD"
                }, {
                    "label": "GBP",
                    "value": "GBP"
                }, {
                    "label": "AUD",
                    "value": "AUD"
                }, {
                    "label": "DKK",
                    "value": "DKK"
                }, {
                    "label": "SEK",
                    "value": "SEK"
                }
                ]
            }, {
                "name": "shipping",
                "type": "currency",
                "label": "webs.module.paypal.field.shipping.label",
                "hint": "webs.module.paypal.field.shipping.hint",
                "currency_ref": "price"
            }, {
                "name": "tax",
                "type": "number",
                "label": "webs.module.paypal.field.tax.label",
                "hint": "webs.module.paypal.field.tax.hint",
                "percentage": true
            }, {
                "name": "button_style",
                "type": "menu",
                "label": "webs.module.paypal.field.buttonstyle.label",
                "hint": "",
                "options": [{
                    "label": "webs.module.paypal.field.buttonstyle.opt.small",
                    "value": "small"
                }, {
                    "label": "webs.module.paypal.field.buttonstyle.opt.large",
                    "value": "large"
                }, {
                    "label": "webs.module.paypal.field.buttonstyle.opt.largecc",
                    "value": "large_cc"
                }
                ]
            }
            ]
        };
        extend.popoverConfig['title'] = translate('webs.module.paypal.title');
        extend.popoverConfig['header'] = translate('webs.module.paypal.header');
        extend.popoverConfig['button_text'] = translate('webs.module.paypal.buttontext');
        extend.popoverConfig['field_defs'][0]['label'] = translate('webs.module.paypal.field.username.label');
        extend.popoverConfig['field_defs'][0]['hint'] = translate('webs.module.paypal.field.username.hint');
        extend.popoverConfig['field_defs'][0]['validation']['message'] = translate('webs.module.paypal.field.username.validationmsg');
        extend.popoverConfig['field_defs'][1]['label'] = translate('webs.module.paypal.field.item.label');
        extend.popoverConfig['field_defs'][1]['hint'] = translate('webs.module.paypal.field.item.hint');
        extend.popoverConfig['field_defs'][1]['validation']['message'] = translate('webs.module.paypal.field.item.validationmsg');
        extend.popoverConfig['field_defs'][2]['label'] = translate('webs.module.paypal.field.price.label');
        extend.popoverConfig['field_defs'][2]['hint'] = translate('webs.module.paypal.field.price.hint');
        extend.popoverConfig['field_defs'][3]['label'] = translate('webs.module.paypal.field.shipping.label');
        extend.popoverConfig['field_defs'][3]['hint'] = translate('webs.module.paypal.field.shipping.hint');
        extend.popoverConfig['field_defs'][4]['label'] = translate('webs.module.paypal.field.tax.label');
        extend.popoverConfig['field_defs'][4]['hint'] = translate('webs.module.paypal.field.tax.hint');
        extend.popoverConfig['field_defs'][5]['label'] = translate('webs.module.paypal.field.buttonstyle.label');
        extend.popoverConfig['field_defs'][5]['options'][0]['label'] = translate('webs.module.paypal.field.buttonstyle.opt.small');
        extend.popoverConfig['field_defs'][5]['options'][1]['label'] = translate('webs.module.paypal.field.buttonstyle.opt.large');
        extend.popoverConfig['field_defs'][5]['options'][2]['label'] = translate('webs.module.paypal.field.buttonstyle.opt.largecc');


        // Module HTML Template
        extend.template = '	{{#case button_style "small"}}\
        <img name="submit" border="0" src="https://www.paypal.com/{{locale}}/i/btn/btn_buynow_SM.gif" alt="{{#l}}webs.module.paypal.slogan{{/l}}"/>\
        {{/case}}\
        \
        {{#case button_style "large"}}\
        <img name="submit" border="0" src="https://www.paypal.com/{{locale}}/i/btn/btn_buynow_LG.gif" alt="{{#l}}webs.module.paypal.slogan{{/l}}"/>\
        {{/case}}\
        \
        {{#case button_style "large_cc"}}\
        <img name="submit" border="0" src="https://www.paypal.com/{{locale}}/i/btn/btn_buynowCC_LG.gif" alt="{{#l}}webs.module.paypal.slogan{{/l}}"/>\
        {{/case}}';

        // Module Default Data
        extend.defaultData = {
            "paypal_username": "",
            "item_name": "",
            "price": "0",
            "price_currency": "",
            "shipping": "0",
            "tax": "0",
            "button_style": "large_cc",
            "locale": "en_US"
        };

        // Module Styles
        extend.styles = {};
        extend.defaultStyle = extend.styles[''];

        module.getMinWidth = function() {
            return 250;
        };

        // Register Toolbars
        bldr.toolbar.register('paypal', {
            "icon": "paypal",
            "items": [{
                "type": "button",
                "style": "blue",
                "label": "Edit Button",
                "name": "settingsDialog"
            }
            ]
        });

        // View JS
        if (!extend.defaultData) {
            extend.defaultData = {
                "locale": "en_US"
            };
        }

        if (typeof(webs) !== 'undefined' && webs.locale) {
            var languageMap = {
                "da-DK": "da_DK",
                "de-DE": "de_DE",
                "en-AU": "en_AU",
                "en-GB": "en_GB",
                "en-US": "en_US",
                "es-ES": "es_ES",
                "fr-CA": "fr_CA",
                "fr-FR": "fr_FR",
                "it-IT": "it_IT",
                "ja-JP": "ja_JP",
                "nl-NL": "nl_NL",
                "no-NO": "no_NO",
                "pl-PL": "pl_PL",
                "sv-SE": "sv_SE",
                "tr-TR": "tr_TR",
                "zh-CN": "zh_CN",
                "zh-TW": "zh_TW",
                "zz-ZZ": "zh_TW"
            };

            extend.defaultData.locale = languageMap[webs.locale] || "en_US";
        }

        // Edit JS
        module.safeOneLoaded = function() {
            this.translateDefaults();
            var self = this;
            setTimeout(function() {
                self.dirty();
            }, 0);
        };

        module.translateDefaults = function() {
            if (this.data.price_currency == "") {
                this.data.price_currency = translate("webs.module.paypal.default_currency");
            }
        };

    })(mod, ext);

    return ModuleClassLoader.register('paypal', mod, ext);

});


// sb2_paragraph Module (custom)

define('module/sb2_paragraph', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader'],
function($, ModuleClassLoader) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="w-sb2_paragraph">\
        {{#if html}}\
        {{{html}}}\
        {{else}}\
        <div class="bldr-placeholder-element"></div>\
        {{/if}}\
        </div>';

        // Module Default Data
        extend.defaultData = {
            "html": ""
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "base": {
                "global": {
                    "css": "view.less"
                },
                "default": true,
                "slug": "base"
            }
        };
        if (!extend.styles['base']['global']) {
            extend.styles['base']['global'] = {};
        }
        extend.styles['base']['global']['js'] = null;
        extend.defaultStyle = extend.styles['base'];

        // Register Toolbars
        bldr.toolbar.register('sb2_paragraph', {
            "icon": "sb2_paragraph",
            "items": [{
                "type": "button",
                "style": "blue",
                "label": "Edit Paragraph",
                "name": "html",
                "dialog": {
                    "url": "blank.html",
                    "heading": "Edit Content Box",
                    "width": 800,
                    "height": 550
                }
            }
            ]
        });

        // View JS


        // Edit JS
        module.onActivate = function() {
            var self = this;
            this.bind("rendered", function() {
                self.attachControls();
            });
        };

        module.onDeactivate = function() {
            this.unbind("rendered");
        };

        module.showSettingsDialog = function() {
            this.toolbar.click("html");
        };

        module.attachControls = function() {
            var
            self = this,
            $ = bldr.pageController.$,
            data = self.data,
            container = self.el,
            moduleId = self.getModuleController().model.id,
            popover;

            // set dialog with the correct URL (moduleId) for editing
            self.toolbar = bldr.toolbar.create("sb2_paragraph");
            popover = self.toolbar.getDialog("html");
            popover.url = "/s/sitebuilder/legacy/editParagraph?moduleId=" + moduleId;
            self.toolbar.show({
                html: self.data.html || "" 
            });

            self.toolbar.addListener({
                html: function(data) {
                    self.data.html = data;
                    self.render();
                }
            });

            container.find(".w-sb2_paragraph").bind("dblclick", function(e) {
                self.toolbar.click("html");
                e.preventDefault();
                return false;
            });
        };


        module.render = function() {
            var self = this;

            if (!this.data.html) {
                this.data.html = "";
                this.dirty();
            }
            if (!this.$html) 
                this.$html = $("html");

            this.el.html(this.html());
            this.renderedHTML = this.data.html;
            this.trigger("rendered");
        };

        module.safeOneLoaded = function() {
            var self = this;
            this.bind("rendered", function() {
                self.el.find("a").data("disableLink", true);
                self.el.delegate("a", "click", function(e) {
                    e.preventDefault();
                    return false;
                });
            });
        };


    })(mod, ext);

    return ModuleClassLoader.register('sb2_paragraph', mod, ext);
});




// audio Module (custom)

define('module/audio', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader'],
function($, ModuleClassLoader) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="webs-audio-wrapper webs-audio-size-{{playerSize}} webs-audio-scheme-{{scheme}}">\
        <div class="colorSwatch"></div>\
        {{#case playerSize "large"}}\
        <div class="webs-audio-highlight"></div>\
        <div class="play-pause"><div class="play-pause-icon"></div></div>\
        <div class="id3-container">\
        <div class="title">{{#if audio}}{{audio.title}}{{/if}}</div>\
        <div class="album">{{#if audio}}{{#if audio.artist}}{{audio.artist}}{{/if}}{{#if audio.album}} ({{audio.album}}){{/if}}{{/if}}</div>\
        </div>\
        <div class="right-controls">\
        <div class="volume-controls">\
        <div class="mute-toggle"></div>\
        <div class="volume-bar-container">\
        <div class="volume-bar volume-1 active"></div><div class="volume-bar volume-2 active"></div><div class="volume-bar volume-3 active"></div><div class="volume-bar volume-4 active"></div><div class="volume-bar volume-5"></div>\
        </div>\
        </div>\
        <div class="time-display">\
        <span class="played">0:00</span>/<span class="total">{{#if audio}}{{formatDuration audio.trackLength}}{{else}}0:00{{/if}}</span>\
        </div>\
        </div>\
        <div class="progress-container">\
        <div class="seekable">\
        <div class="current"></div>\
        </div>\
        </div>\
        {{/case}}\
        {{#case playerSize "small"}}\
        <div class="play-pause"></div>\
        <div class="id3-container">\
        <div class="title">{{#if audio}}{{audio.title}}{{/if}}</div>\
        <div class="album">{{#if audio}}{{#if audio.artist}}{{audio.artist}}{{/if}}{{#if audio.album}} ({{audio.album}}){{/if}}{{/if}}</div>\
        <div class="tip"></div>\
        </div>\
        <div class="progress-crop">\
        <div class="progress-container">\
        <div class="seekable">\
        <div class="current"></div>\
        </div>\
        </div>\
        </div>\
        <div class="right-controls">\
        <div class="time-display">\
        <span class="remaining">-{{#if audio}}{{formatDuration audio.trackLength}}{{else}}0:00{{/if}}</span>\
        </div>\
        <div class="volume-controls">\
        <div class="mute-toggle"></div>\
        <div class="volume-bar-container">\
        <div class="volume-bar volume-1 active"></div><div class="volume-bar volume-2 active"></div><div class="volume-bar volume-3 active"></div><div class="volume-bar volume-4 active"></div><div class="volume-bar volume-5"></div>\
        </div>\
        </div>\
        </div>\
        {{/case}}\
        <div class="jPlayer"></div>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "playerSize": "large",
            "scheme": "light",
            "color": null,
            "autoplay": false
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "instance": {
                    "css": "view.each.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars
        bldr.toolbar.register('audio', {
            "icon": "audio",
            "items": [{
                "label": "Upload Audio",
                "name": "uploadAudio",
                "style": "blue",
                "dialog": {
                    "url": "/s/modules/audio/v1.14/audio.dialog.html",
                    "localizeHeading": "webs.module.audio.dialogHeading",
                    "width": 690,
                    "height": 400
                }
            }, {
                "type": "menu",
                "label": "Settings",
                "menuLabel": "Advanced Settings",
                "menuPosition": "left",
                "menuClassNames": "w-audio-advanced",
                "menu": [{
                    "type": "dropdown",
                    "label": "Size",
                    "name": "playerSize",
                    "display": "inline",
                    "width": 80,
                    "options": [{
                        "label": "Large",
                        "value": "large"
                    }, {
                        "label": "Small",
                        "value": "small"
                    }
                    ]
                }, {
                    "type": "dropdown",
                    "label": "Scheme",
                    "name": "scheme",
                    "display": "inline",
                    "width": 80,
                    "options": [{
                        "label": "Light",
                        "value": "light"
                    }, {
                        "label": "Dark",
                        "value": "dark"
                    }
                    ]
                }, {
                    "type": "colorpicker-menu",
                    "label": "Color",
                    "name": "color",
                    "display": "inline",
                    "delayUpdate": true
                }, {
                    "type": "switch-bare",
                    "label": "Autoplay",
                    "name": "autoplay",
                    "display": "inline"
                }
                ]
            }
            ]
        });

        // View JS
        module.oneLoaded = function audio_oneLoaded() {
            if (typeof window.top.dsnr != "undefined") {
                this.el.find(".seekable").width("50%");
                this.el.find(".current").width("50%");
                return;
            }
            var self = this;
            this.player = this.el.find(".jPlayer");
            require(['jquery.jplayer'], function() {
                jQuery.jPlayer.timeFormat.padMin = false;
                self.player.jPlayer({
                    supplied: "mp3",
                    swfPath: webs.props.globalAssetBaseURL + '/static/global/js/jquery/plugins/jplayer2/Jplayer.swf',
                    volume: 0.8,
                    solution: "flash, html",
                    ready: function jPlayer_ready() {
                        var path = self.data.audio.webs.path,
                        url;
                        if (path.indexOf('/') == 0) 
                            path = path.substring(1);
                        url = webs.site.url + path;
                        self.player.jPlayer("setMedia", {
                            "mp3": url
                        });
                        self.constructPlayer();
                        if (self.data.autoplay) 
                            self.togglePlaying();
                    }
                });

            });
        };

        module.constructPlayer = function audio_constructPlayer() {
            var self = this,
            seekable = self.el.find(".seekable"),
            current = self.el.find(".current"),
            played = self.el.find(".played"),
            total = self.el.find(".total"),
            remaining = self.el.find(".remaining");
            this.playing = false;
            this.el.find(".play-pause").click(function clickPlay() {
                self.togglePlaying();
            });
            function timeupdate(event) {
                seekable.width((self.el.find(".progress-container").width() * event.jPlayer.status.seekPercent / 100 - (self.data.playerSize == "large" ? 6 : 0)) + "px");
                current.width(event.jPlayer.status.currentPercentRelative + "%");
                played.text(jQuery.jPlayer.convertTime(event.jPlayer.status.currentTime));
                total.text(jQuery.jPlayer.convertTime(event.jPlayer.status.duration));
                remaining.text("-" + jQuery.jPlayer.convertTime(event.jPlayer.status.duration - event.jPlayer.status.currentTime))
            }
            this.player.bind(jQuery.jPlayer.event.timeupdate, timeupdate);
            this.player.bind(jQuery.jPlayer.event.progress, timeupdate);
            this.player.bind(jQuery.jPlayer.event.ended, function playEnded(event) {
                self.player.jPlayer("playHead", 0);
                self.playing = false;
                self.el.find(".play-pause").removeClass("playing");
            });
            seekable.click(function seekClick(event) {
                self.player.jPlayer("playHead", event.offsetX / seekable.width() * 100);
            });
            this.player.bind(jQuery.jPlayer.event.play, function play(event) {
                self.el.find(".play-pause").addClass("playing");
                self.playing = true;
            });
            this.player.bind(jQuery.jPlayer.event.pause, function pause(event) {
                self.el.find(".play-pause").removeClass("playing");
                self.playing = false;
            });
            var muteToggle = this.el.find(".mute-toggle");
            self.muted = false;
            muteToggle.click(function toggleMute() {
                self.player.jPlayer(self.muted ? "unmute" : "mute");
                self.muted = !self.muted;
                muteToggle[self.muted ? "addClass" : "removeClass"]("muted");
            });
            this.el.find(".volume-bar-container").click(function changeVolume(event) {
                var volume = (event.pageX - self.el.find(".volume-bar-container").offset().left) / self.el.find(".volume-bar-container").width();
                volume = Math.ceil(volume * 5) / 5;
                self.player.jPlayer("volume", volume);
            });
            this.player.bind(jQuery.jPlayer.event.volumechange, function volumeChanged(event) {
                var volume = event.jPlayer.options.volume;
                for (i = 1; i <= 5; ++i)
                    self.el.find(".volume-" + i)[(i / 5 <= volume) ? "addClass" : "removeClass"]("active");
            });
        };

        module.togglePlaying = function audio_togglePlaying() {
            if (this.playing) {
                this.player.jPlayer("pause");
            } else {
                this.player.jPlayer("play");
            }
        };

        // Edit JS
        module.safeOneLoaded = function() {
            var self = this;
            this.bind("rendered", function rendered() {
                self.el.find(".seekable").width("50%");
                self.el.find(".current").width("50%");
                self.updateInstanceLess();
            });
        };

        module.edit = function audio_edit() {
            var self = this;

            this.toolbar = bldr.toolbar.create('audio');
            this.toolbar.bind({
                "uploadAudio": function uploadAudio(val) {
                    self.data.audio = val;

                    if (self.data.audio.title == null) {
                        var filename = self.data.audio.url;
                        filename = filename.substring(filename.lastIndexOf('/') + 1);
                        filename = filename.substring(0, filename.lastIndexOf('.'));
                        var dashIndex = filename.indexOf('-');
                        if (dashIndex > 0) {
                            self.data.audio.artist = filename.substring(0, dashIndex).replace(/-|\s$/, '');
                            self.data.audio.title = $.trim(filename.substring(dashIndex)).replace(/^-|\s/, '');
                        } else {
                            self.data.audio.title = filename;
                        }
                    }

                    self.render();
                },
                "playerSize": function updatePlayerSize(val) {
                    self.data.playerSize = val;
                    self.render();
                },
                "scheme": function updateScheme(val) {
                    self.data.scheme = val;
                    self.render();
                },
                "color": function updateColor(val) {
                    self.data.color = val;
                    self.updateInstanceLess();
                },
                "autoplay": function updateAutoplay(val) {
                    self.data.autoplay = val;
                    // self.render();
                },
            });
            var color = this.data.color || this.el.find(".colorSwatch").css("background-color");
            this.toolbar.show({
                "uploadAudio": this.data.audio || {},
                "playerSize": this.data.playerSize,
                "scheme": this.data.scheme,
                "color": color,
                "autoplay": this.data.autoplay
            });
        };

        module.attachControls = function audio_attachControls() {
            var self = this;
            this.el.find(".webs-audio-wrapper").bind("dblclick", function(e) {
                self.toolbar.click("uploadAudio");
                e.preventDefault();
                return false;
            });
        };

        module.getMinWidth = function() {
            return 200;
        };

    })(mod, ext);

    return ModuleClassLoader.register('audio', mod, ext);
});




// table Module (custom)

define('module/table', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.table'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return true;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="webs-table-wrap">\
        <div class="webs-table-container">\
        <table class="webs-table {{#case showBorder true}}showborder{{/case}} {{#case showBorder false}}hideborder{{/case}}">\
        <thead {{#unless showHeader}}class="disabled"{{/unless}}>\
        <tr>{{#each header}}<td> <div class="webs-table-content webs-text">{{#if this}}{{{this}}}{{else}}{{#l}}webs.module.table.headerPlaceholder{{/l}}{{/if}}</div> </td>{{/each}}</tr>\
        </thead>\
        <tbody>\
        {{#each_object rows}}\
        <tr class="{{#odd _i_}}odd{{else}}even{{/odd}}">\
        {{#each_object val}}\
        <td {{#unless ../_i_}}style="width: {{array_value ../../../widths _i_}}%;"{{/unless}}>\
        <div class="webs-table-content webs-text"> {{#if val}}{{{val}}}{{else}}{{#l}}webs.module.table.cellPlaceholder{{/l}}{{/if}} </div>\
        </td>\
        {{/each_object}}\
        </tr>\
        {{/each_object}}\
        </tbody>\
        </table>\
        </div>\
        </div>';

        // Module Default Data
        extend.defaultData = {
            "color_header": null,
            "color_row": null,
            "color_altrow": null,
            "color_border": null,
            "showHeader": true,
            "showBorder": null,
            "header": [""],
            "widths": [100],
            "rows": [[""]],
            "linksData": {}
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less",
                    "js": "view.js"
                },
                "instance": {
                    "css": "view.each.less"
                },
                "slug": "default"
            }
        };
        extend.styles['default']['global']['js'] = (function(module, extend) {


            /* view.js */

            return module;
        });
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS


        // Edit JS


        var $$ = bldr.pageController.$;
        $.extend(module, {
            onActivate: function() {
                this.$table = this.el.find(".webs-table");
                this.$tbody = this.$table.find("tbody");
                this.$thead = this.$table.find("thead");

                this.$table.addClass("bldr-table-real");

                this.cellCtrl = TableCellController.init({
                    el: this.el.find(".webs-table-container"),
                    $table: this.$table,
                    $thead: this.$thead,
                    $tbody: this.$tbody,
                    m: this,
                    colors: this.data.colors,
                    linksData: this.data.linksData
                });
                this.cellCtrl.bind("deleteRow", $.proxy(this, "deleteRow"));
                this.cellCtrl.bind("deleteCol", $.proxy(this, "deleteCol"));
                this.cellCtrl.bind("swapRow", $.proxy(this, "swapRow"));
                this.cellCtrl.bind("swapCol", $.proxy(this, "swapCol"));
                this.cellCtrl.bind("setWidth", $.proxy(this, "setWidth"));
                this.cellCtrl.bind("heightChanged", $.proxy(this, "refreshCover"));

                this.createModulebar();

                this.trigger("rowChanged");
                this.setAllWidths();
            },

            saveAllCellContents: function() {
                var $trs = this.$tbody.children(), $tds, data = this.data;

                $trs.each(function(i) {
                    $tds = $trs.eq(i).find(".webs-table-content");
                    $tds.each(function(j) {
                        data.rows[i][j] = $tds.eq(j).html() || "&nbsp;";
                    });
                });

                $tds = this.$thead.find(".webs-table-content");
                $tds.each(function(i) {
                    data.header[i] = $tds.eq(i).html() || "&nbsp;";
                });
                this.dirty();
            },

            refreshCover: function() {
                bldr.pageController.showCover();
            },

            toggleHeader: function(v) {
                this.data.showHeader = v;
                this.cellCtrl.toggleHeader(v);
                this.$headerColorContainer[v ? "show" : "hide"]();
                this.dirty();
            },

            toggleBorder: function(v) {
                this.data.showBorder = v;
                if (v) {
                    this.$borderColorContainer.show();
                    this.$table.removeClass("hideborder").addClass("showborder");
                } else {
                    this.$borderColorContainer.hide();
                    this.$table.removeClass("showborder").addClass("hideborder");
                }
                this.dirty();
            },

            setColor: function(hex, colorType) {
                this.data[colorType] = hex;
                this.trigger("colorChanged");
            },

            setCellContent: function(content, header, row, col) {
                if (header) {
                    this.data.header[col] = content;
                } else if (this.data.rows[row]) {
                    this.data.rows[row][col] = content;
                }
            },

            changeRowSize: function(i) {
                var rowSize = this.getRowSize();
                if (i > rowSize) 
                    this.addRow();
                else if (i < rowSize) 
                    this.deleteRow(rowSize - 1);
                this.cellCtrl.trigger("hide");
            },

            changeColSize: function(i) {
                var colSize = this.getColSize();
                if (i > colSize) 
                    this.addCol();
                else if (i < colSize) 
                    this.deleteCol(colSize - 1);
                this.cellCtrl.trigger("hide");
            },

            getRowSize: function() {
                return this.data.rows.length;
            },
            getColSize: function() {
                return this.data.rows[0].length;
            },

            addRow: function() {
                var dim = {
                    colSize: this.getColSize(),
                    rowSize: this.getRowSize()
                }, newRow = [];
                for (var i = 0; i < dim.colSize; i++) 
                    newRow.push("&nbsp;");
                this.data.rows.push(newRow);
                this.cellCtrl.addRow(dim);
                this.refreshCover();
            },

            addCol: function() {
                this.data.header.push("&nbsp;");
                for (var i = 0, r = this.data.rows, l = r.length; i < l; i++) 
                    r[i].push("&nbsp;");
                this.cellCtrl.addCol();
                this.setAllWidths(true);
            },


            canDeleteRow: function(i) {
                return this.data.rows.length > 1;
            },
            canDeleteCol: function(i) {
                return this.data.rows[0].length > 1;
            },

            deleteRow: function(i) {
                if (this.canDeleteRow()) {
                    this.data.rows.splice(i, 1);
                    this.cellCtrl.deleteRow(i);
                    this.trigger("rowChanged");
                    this.refreshCover();
                }
            },

            deleteCol: function(i) {
                if (this.canDeleteCol()) {
                    this.data.header.splice(i, 1);

                    for (var j = 0, len = this.data.rows.length; j < len; j++) {
                        this.data.rows[j].splice(i, 1);
                    }
                    this.cellCtrl.deleteCol(i);
                    this.setAllWidths(true);
                }
            },

            swapArray: function(array, i, j) {
                array[i] = array.splice(j, 1, array[i])[0];
            },

            swapCol: function(i, j) {
                this.swapArray(this.data.header, i, j);
                this.swapArray(this.data.widths, i, j);
                this.cellCtrl.setWidth(i, this.data.widths[i]);
                this.cellCtrl.setWidth(j, this.data.widths[j]);
                for (var r = 0; r < this.data.rows.length; r++) {
                    this.swapArray(this.data.rows[r], i, j);
                }

                this.cellCtrl.swapCol(i, j);
            },

            swapRow: function(i, j) {
                this.swapArray(this.data.rows, i, j);
                this.cellCtrl.swapRow(i, j);
                this.trigger("rowChanged");
                if (i === 0 || j === 0) 
                    this.applyWidthToColumns();
            },


            /**
            	 * Sets width on current cell, make sure the cell to the right is adjusted properly
            	 */
            setWidth: function(col, width, setDom) {
                var $cell, totalWidth = this.data.widths[col] + this.data.widths[col + 1];
                this.data.widths[col] = width;
                this.data.widths[col + 1] = totalWidth - width;
                if (setDom) {
                    this.cellCtrl.setWidth(col, width);
                    this.cellCtrl.setWidth(col + 1, totalWidth - width);
                }
            },

            /**
            	 * In case a width is missing, make sure all cells receive widths that all up to 100
            	 */
            setAllWidths: function(force) {
                if (this.getTotalWidth() !== 100 || force) {
                    var total = this.cellCtrl.getTotalWidthPixels(), $cells = this.cellCtrl.getRowCells(0);
                    for (var i = 0, len = this.data.rows[0].length; i < len; i++) {
                        this.data.widths[i] = 100 * $cells.eq(i).width() / total;
                    }
                }
            },

            /**
            	 * Total width inside data, should add up to 100%
            	 */
            getTotalWidth: function() {
                var totalWidth = 0;
                for (var i = 0, len = this.data.rows[0].length; i < len; i++) 
                    totalWidth += this.data.widths[i];
                return totalWidth;
            },

            applyWidthToColumns: function() {
                var widths = this.data.widths, $firstRowCells = this.cellCtrl.getRowCells(0);
                this.$table.find("td").width("");
                $firstRowCells.each(function(i) {
                    $firstRowCells.eq(i).width(widths[i] + "%");
                });
            },

            getData: function() {
                return this.data;
            },
            destroy: function() {
                this.trigger("destroy");
            },

            safeOneLoaded: function() {
                this.bind("rendered", $.proxy(this, "updateInstanceLess"));
            },

            settingsClick: function() {
                this.cellCtrl.hideControls();
            },

            fetchStyleColors: function() {
                var tempTable = $('<table class="webs-table">' +
                '<thead><tr class="odd"><td></td><td></td></tr></thead>' +
                '<tbody><tr class="even"><td></td><td></td></tr></tbody></table>').insertAfter(this.$table);
                this.initialStyleColors = {
                    'color_header': this.rgbToHex(tempTable.find("thead td").css("background-color")),
                    'color_row': this.rgbToHex(tempTable.find("tr.odd td").css("background-color")),
                    'color_altrow': this.rgbToHex(tempTable.find("tr.even td").css("background-color")),
                    'color_border': this.rgbToHex(tempTable.css("border-color"))
                };
                tempTable.remove();
            },

            rgbToHex: function(string) {
                var match, r, g, b;
                if (typeof string === "string") {
                    if (string === "" || string.match("transparent")) 
                        string = "rgba(255, 255, 255, 0)";
                    match = string.match(/(rgb|rgba)\(([0-9]{1,3}),\s*([0-9]{1,3}),\s*([0-9]{1,3})/);
                    if (match) {
                        r = parseInt(match[2], 10).toString(16);
                        g = parseInt(match[3], 10).toString(16);
                        b = parseInt(match[4], 10).toString(16);
                        if (r.length === 1) 
                            r = "0" + r;
                        if (g.length === 1) 
                            g = "0" + g;
                        if (b.length === 1) 
                            b = "0" + b;
                        string = "#" + r + g + b;
                    }
                }
                return string;
            },

            createModulebarMenuColor: function(colorName, label) {
                this.moduleBar.bind(colorName, $.proxy(this, "setColor"));
                return {
                    display: "inline",
                    type: "colorpicker-menu",
                    name: colorName,
                    label: label,
                    value: this.data[colorName] || this.initialStyleColors[colorName],
                    labelWidth: 100
                };
            },

            createModulebarToggle: function(datakey, label, method) {
                this.moduleBar.bind(datakey, $.proxy(this, method || datakey));
                return {
                    display: 'inline',
                    type: 'switch-bare',
                    label: label,
                    name: datakey,
                    classNames: this.data[datakey] ? 'active' : null,
                    labelWidth: 100
                };
            },

            createModulebar: function() {
                var data = this.data, menu = [], addRowBtn, addColBtn;

                // If showBorder isn't set, try to fetch the value
                if (typeof data.showBorder === "undefined" || data.showBorder === null) {
                    var borderWidth = parseInt(this.$table.css("border-width"), 10);
                    var borderStyle = this.$table.css("border-style");
                    data.showBorder = !!(borderWidth && borderStyle !== "none");
                }

                this.fetchStyleColors();

                this.moduleBar = bldr.pageController.getModuleBar();
                this.$modulebar = this.moduleBar.container.addClass('bldr-modulebar-bottom');
                this.$modulebar.delegate(".bldr-modulebar-options", "click", $.proxy(this, "settingsClick"));

                menu.push(this.createModulebarToggle("showHeader", "header", "toggleHeader"));
                menu.push(this.createModulebarMenuColor("color_header", "headerColor"));
                menu.push(this.createModulebarToggle("showBorder", "border", "toggleBorder"));
                menu.push(this.createModulebarMenuColor("color_border", "borderColor"));
                menu.push(this.createModulebarMenuColor("color_row", "rowColor"));
                menu.push(this.createModulebarMenuColor("color_altrow", "alternateRow"));

                this.moduleBar.build({
                    module: this,
                    width: 210,
                    title: "Table Settings",
                    menu: menu
                });

                // modulebar min-widths -> with button icons: 160px, with just text: 220px
                // if less than 220, use icons instead of text
                addColBtn = {
                    style: 'blue',
                    label: 'addColumn',
                    name: 'addCol',
                    _icon: 'table-addcol' 
                };
                addRowBtn = {
                    style: 'blue',
                    label: 'addRow',
                    name: 'addRow',
                    _icon: 'table-addrow' 
                };
                if (this.el.parent().width() < 220) {
                    addRowBtn.icon = 'table-addrow';
                    addColBtn.icon = 'table-addcol';
                    this.$modulebar.css("min-width", 160);
                } else {
                    this.$modulebar.css("min-width", 220);
                }
                this.moduleBar.addButton(addColBtn);
                this.moduleBar.addButton(addRowBtn);
                this.moduleBar.bind({
                    "addCol": $.proxy(this, "addCol"),
                    "addRow": $.proxy(this, "addRow")
                });


                this.$settings = this.$modulebar.find(".bldr-modulebar-options").addClass("w-more-bottom");
                this.$headerColorContainer = this.$settings.find(".w-tbui-color_header").parent();
                this.$borderColorContainer = this.$settings.find(".w-tbui-color_border").parent();

                this.toggleBorder(data.showBorder);
                this.toggleHeader(data.showHeader);
            }
        });




        /**
         * Handles cell event bindings, holds resizer, move, delete controllers
         * deals with all DOM-related operations
         */
        var TableCellController = Spine.Controller.create({
            events: {
                "mouseleave .bldr-table-real .webs-table-container": "tableLeave",
                "mouseenter .bldr-table-real td": "cellEnter",
                "mousedown .bldr-table-real td": "cellMousedown",
                "mouseup .bldr-table-real td": "cellMouseup"
            },

            cellMousedown: function(e) {
                // if performing mousedown on the same cell, don't do anything
                if (this.activeText.editingCell === e.currentTarget) 
                    return;

                // save previous editing text
                this.saveActiveText();

                this.createTextControls(e.currentTarget);
            },

            // When we first activate, rely on mouseup to set active text cell
            cellMouseup: function(e) {
                if (!this.activeText) {
                    this.activeText = {};
                    this.cellMousedown(e);
                }
            },

            cellEnter: function(e) {
                this.show(e.currentTarget);
            },
            tableLeave: function() {
                this.hide();
            },

            init: function() {
                var initParam = {
                    el: this.el,
                    cellCtrl: this
                };
                this.leftResizer = LeftCellResizer.init(initParam);
                this.rightResizer = RightCellResizer.init(initParam);
                this.rowBar = RowTitlebar.init(initParam);
                this.colBar = ColTitlebar.init(initParam);

                this.elWidth = this.el.width();

                this.$table.find(".webs-table-content").attr("contenteditable", true);

                this.m.bind("colorChanged", $.proxy(this, "applyColorStyle"));
                this.m.bind("rowChanged", $.proxy(this, "assignAltClass"));
                this.m.bind("destroy", $.proxy(this, "destroy"));

                this.tableHeight = this.$table.height();
                this.checkHeightInterval = window.setInterval($.proxy(this, "checkTableHeight"), 500);
            },

            checkTableHeight: function() {
                var height = this.$table.height();
                if (height !== this.tableHeight) {
                    this.tableHeight = height;
                    this.trigger("heightChanged");
                }
            },

            createTextControls: function(cell) {
                var $cell = $$(cell), $el;
                var controls = $cell.data("controls");
                if (!controls) {
                    $el = $cell.children(".webs-table-content");
                    controls = bldr.controls.create('text', {
                        el: $el,
                        html: $el.html(),
                        linksData: this.linksData
                    });
                    controls.show();
                    $cell.data("controls", controls);
                }
                this.activeText = {
                    controls: controls,
                    editingCell: cell,
                    $cell: $cell
                };
            },

            saveActiveText: function() {
                var $cell, row, col, header, controls = this.activeText && this.activeText.controls;
                if (controls) {
                    if ((!controls.getData() || !controls.getData().html) && controls.wizzy) 
                        controls.wizzy.$el.html("&nbsp;");
                    $cell = this.activeText.$cell;
                    row = $cell.parent().index();
                    col = $cell.index();
                    header = this.$thead.find($cell)[0];
                    this.m.setCellContent(controls.getData().html, header, row, col);

                    controls.destroy();
                    this.activeText.controls = null;
                    $cell.data("controls", "");
                    this.m.dirty();
                }
            },

            applyColorStyle: function() {
                this.m.updateInstanceLess();
            },

            assignAltClass: function() {
                this.getOddRows().removeClass("even").addClass("odd");
                this.getEvenRows().removeClass("odd").addClass("even");
            },

            getCellTemplate: function() {
                return '<td><div class="webs-table-content webs-text" contenteditable="true">&nbsp;</div></td>';
            },

            toggleHeader: function(v) {
                this.$thead[v ? "removeClass" : "addClass"]("disabled");
                this.trigger("heightChanged");
            },

            addRow: function(dim) {
                var tr = '<tr class="' + (dim.rowSize%2 ? 'odd' : 'even') + '">';
                for (var i = 0; i < dim.colSize; i++) 
                    tr += this.getCellTemplate();
                this.$tbody.append(tr + '</tr>');
            },
            addCol: function() {
                this.$table.find("tr").append(this.getCellTemplate());
            },

            deleteRow: function(i) {
                var $cell, $row = this.$table.find("tbody > tr").eq(i).remove();
                $row.children().each(function(i, td) {
                    $cell = $$(td);
                    if ($cell.data("controls")) 
                        $cell.data("controls").destroy();
                });
            },

            deleteCol: function(i) {
                var $cell, $rows = this.$table.find("tbody > tr");
                this.destroyTextControls(this.$table.find("thead td").eq(i).remove());
                for (var j = 0, len = $rows.length; j < len; j++) {
                    this.destroyTextControls($rows.eq(j).children().eq(i).remove());
                }
            },

            destroyTextControls: function(td) {
                var $cell = $$(td), controls = $cell.data("controls");
                if (controls) {
                    controls.destroy();
                    $cell.data("controls", null);
                }
            },

            swapCol: function(i, j) {
                this.swapCells(this.getColCells(i), this.getColCells(j));
            },
            swapRow: function(i, j) {
                this.swapCells(this.getRowCells(i), this.getRowCells(j));
            },

            swapCells: function($cells1, $cells2) {
                var $i, $j, html, width;
                $cells1.each(function(i) {
                    $i = $cells1.eq(i).children(".webs-table-content");
                    $j = $cells2.eq(i).children(".webs-table-content");
                    html = $j.html();
                    $j.html($i.html());
                    $i.html(html);
                });
            },

            setWidth: function(col, width) {
                this.getRowCells(0).eq(col).width(width + "%");
            },

            heightChanged: function() {
                this.trigger("heightChanged");
            },

            hideControls: function() {
                this.trigger("hide");
            },

            // does not account for padding
            getTableCellsWidth: function() {
                var w = 0, $cells = this.getRowCells();
                $cells.each(function(i) {
                    w += $cells.eq(i).width();
                });
                return w;
            },


            getColPosition: function() {
                return this.getRowCells().index(this.$activeCell[0]);
            },
            getRowPosition: function() {
                return this.$activeRow.parent().children().index(this.$activeRow[0]);
            },
            isFirstColumn: function() {
                return this.getColPosition() === 0;
            },
            isLastColumn: function() {
                return this.getColPosition() === this.getRowCells().length - 1;
            },
            isLastRow: function() {
                return this.getRowPosition() === this.$activeRow.parent().children().length - 1;
            },

            getRowCount: function() {
                return this.$table.find("tbody tr").length;
            },
            getColCount: function() {
                return this.$table.find("tbody tr").eq(0).children().length;
            },

            // get first non-header row
            getOddRows: function() {
                return this.$table.find("tbody tr:nth-child(even)");
            },
            getEvenRows: function() {
                return this.$table.find("tbody tr:nth-child(odd)");
            },

            getColCells: function(i, tbody) {
                var extra = tbody ? "tbody > " : "";
                if (typeof i !== "number") 
                    i = this.getColPosition();
                return this.$table.find(extra + "tr > td:nth-child(" + (i + 1) + ")");
            },
            getRowCells: function(i) {
                if (typeof i !== "number") 
                    return this.$activeRow.children();
                return this.$table.find("tbody > tr").eq(i).children();
            },

            getTableWidth: function() {
                return this.elWidth;
            },
            getTableHeight: function() {
                return this.el.height();
            },

            getTotalWidthPixels: function() {
                var totalWidth = 0, $cells = this.getRowCells(0);
                $cells.each(function(i) {
                    totalWidth += $cells.eq(i).width();
                });
                return totalWidth;
            },

            getCellWidth: function(i) {
                var $cell = typeof i === "number" ? this.$table.find("tbody td").eq(i) : this.$activeCell;
                return $cell.width() + parseInt($cell.css("padding-left"), 10) + parseInt($cell.css("padding-right"), 10);
            },

            // doesn't apply to header cell
            getCellHeight: function(i) {
                var $cell = typeof i === "number" ? this.$table.find("tbody tr:nth-child(" + (i + 1) + ") td").eq(0) : this.$activeCell;
                return $cell.height() + parseInt($cell.css("padding-top"), 10) + parseInt($cell.css("padding-bottom"), 10);
            },

            getColWidth: function(i) {
                return this.getCellWidth(i);
            },
            getRowHeight: function(i) {
                return this.getCellHeight(i);
            },

            getLastCellInRow: function() {
                return this.getRowCells().last();
            },
            getLastCellInCol: function() {
                return this.$table.find("tr").last().children().eq(this.getColPosition());
            },

            deleteCurrentRow: function() {
                var $nextRow = this.$activeRow[this.isLastRow() ? "prev" : "next"]();
                this.trigger("deleteRow", this.getRowPosition());
                $nextRow.children().last().mouseenter();
            },
            deleteCurrentCol: function() {
                var $nextCell = this.$activeCell[this.isLastColumn() ? "prev" : "next"]();
                this.trigger("deleteCol", this.getColPosition());
                $nextCell.mouseenter();
            },

            show: function(cell) {
                this.$activeCell = $(cell);
                this.$activeRow = this.$activeCell.parent();
                if (!this.m.$settings.hasClass("active")) 
                    this.trigger("show", this.$activeCell);
            },
            hide: function() {
                this.trigger("hide");
            },
            destroy: function() {
                this.saveActiveText();
                window.clearInterval(this.checkHeightInterval);
            }
        });



        /**
         * Controller for dragging and deleting a column or row
         */
        var CellTitlebar = Spine.Controller.create({

            init: function() {
                this.$win = bldr.pageController.dom.$win;
                this.$body = this.el.parents("body")
                .mousemove( $.proxy(this, "moveMousemove"))
                .mouseup( $.proxy(this, "moveMouseup"));
                this.$ghostTable = $('<table class="bldr-table-ghost webs-table"/>');
                this.$move = $('<div class="bldr-table-move"/>').mousedown($.proxy(this, "moveMousedown"));
                this.$delete = $('<div class="bldr-table-delete"/>').click($.proxy(this, "deleteClick"));
                this.$bar = $('<div class="bldr-table-titlebar"/>').append(this.$move, this.$delete, this.$ghostTable);
                this.el.append(this.$bar);

                this.$ghostCover = $('<div class="bldr-table-ghostcover"/>');

                this.cellCtrl.bind("hide", $.proxy(this, "hide"));
                this.cellCtrl.bind("show", $.proxy(this, "show"));

                this.onInit();
            },

            // removes a column or row
            deleteClick: function() {},

            // titlebar move bindings
            moveMousedown: function(e) {
                this.moveflag = true;
                this.dragStart(e);
            },
            moveMousemove: function(e) {
                if (this.moveflag) {
                    this.moveGhost(e);
                    return false;
                }
            },
            moveMouseup: function() {
                if (this.moveflag) {
                    this.moveflag = false;
                    this.dragEnd();
                }
            },

            // setup for moving col/row
            dragStart: function(e) {
                var targetPosition;

                this.el.addClass("moving");
                this.$bar.addClass("moving");
                this.$body.addClass("noselect");
                this.$ghostTable.html(this.getGhostContent());


                if (typeof e.offsetX === "undefined" || typeof e.offsetY === "undefined") {
                    targetOffset = $(e.target).offset();
                    e.offsetX = e.pageX - targetOffset.left;
                    e.offsetY = e.pageY - targetOffset.top;
                }

                this.elOffset = this.el.offset();
                this.relOffset = {
                    x: e.offsetX - this.$win.scrollLeft(),
                    y: e.offsetY - this.$win.scrollTop()
                };

                this.storeCellInfo();
                this.showGhostCover();
                if (typeof this.onDragStart === "function") 
                    this.onDragStart(e);
            },

            // cache cell info to use during dragging
            storeCellInfo: function() {
                var axis = this.cssAxis;
                var $cells = this.getAdjacentCells();
                var cellPos = this.cellPositions = [];
                this.currentPosition = this.getPosition();
                this.originalPosition = this.currentPosition;
                $cells.each(function(i) {
                    cellPos.push( $cells.eq(i).position()[axis] );
                });
            },

            // if column titlebar, get all row tds, vice versa
            getAdjacentCells: function() {},

            // gets index of current cell with respect to row/col
            getPosition: function() {},

            // dragging col/row
            moveGhost: function() {},

            // finishes moving a column or row
            dragEnd: function() {
                this.el.removeClass("moving moving-row moving-col");
                this.$bar.removeClass("moving");
                this.$body.removeClass("noselect");
                this.$ghostCover.remove();
                this.$ghostTable.html("");
                this.hide();
            },

            // sets and resets widths on a collection
            setCellWidths: function($collection, reset) {
                $collection.each(function(i, child) {
                    child = $(child);
                    child.width(reset ? "" : child.width());
                });
            },

            // gets content from current col/row so we can build a draggable fake one 
            getGhostContent: function() {},

            // displays a white overlay on where the column or row should be
            showGhostCover: function() {
                this.$ghostCover.css({
                    width: this.getGhostWidth(),
                    height: this.getGhostHeight()
                });
                this.el.append(this.$ghostCover);
                this.setGhostCover(this.currentPosition, false);
            },
            getGhostWidth: function() {},
            getGhostHeight: function() {},

            // animates the width and position fo the ghost cover
            setGhostCover: function(i, animate) {
                if (typeof animate === "undefined") 
                    animate = true;
                var css = {};

                // when moving forward past your original position, account for extra widths
                // just grab the css position
                css[this.cssAxis] = i > this.originalPosition ? this.getCellPosition(i) : this.cellPositions[i];

                if (animate) {
                    this.$ghostCover.stop(0, 0).animate(css, 200);
                } else {
                    this.$ghostCover.css(css);
                }
            },

            // displays titlebar for a given cell
            show: function($cell) {
                if (!this.moveflag) {
                    this.$bar.addClass("active").css({
                        left: 0,
                        top: 0
                    });
                    this.onShow($cell);
                }
            },

            // hides the titlebar
            hide: function() {
                if (!this.moveflag) 
                    this.$bar.removeClass("active");
            }
        }),
        RowTitlebar = CellTitlebar.create({
            onInit: function() {
                this.$bar.addClass("bldr-table-titlerow");
                this.$thead = this.el.find("thead");
                this.cssAxis = "top";
                this.$move.attr("title", translate('webs.module.table.moveRow'));
                this.$delete.attr("title", translate('webs.module.table.deleteRow'));
            },

            deleteClick: function() {
                this.cellCtrl.deleteCurrentRow();
            },

            getGhostContent: function() {
                var $ar = this.cellCtrl.$activeRow, $cells = this.cellCtrl.getRowCells(), content;
                this.setCellWidths($cells);
                content = '<tbody><tr class="' + $ar.attr("class") + '">' + $ar.html() + '</tr></tbody>';
                if (this.cellCtrl.getRowPosition() !== 0) 
                    this.setCellWidths($cells, true);
                return content;
            },

            getAdjacentCells: function() {
                return this.cellCtrl.getColCells(null, true);
            },
            getPosition: function() {
                return this.cellCtrl.getRowPosition();
            },

            getGhostWidth: function() {
                return this.cellCtrl.getTableWidth();
            },
            getGhostHeight: function() {
                return this.cellCtrl.getCellHeight();
            },
            getCellPosition: function(i) {
                return this.cellCtrl.getRowCells(i).eq(0).position().top;
            },

            onDragStart: function(e) {
                this.el.addClass("moving-row");
            },

            moveGhost: function(e) {
                var left = e.clientX - this.elOffset.left + this.relOffset.x;
                var top = e.clientY - this.elOffset.top - this.relOffset.y;
                var handleTop = top + this.cellHeight / 2;
                var cps = this.cellPositions, curr = this.currentPosition;

                this.$bar.css({
                    "top": top,
                    "left": left
                });

                if (cps[curr + 1] && handleTop > cps[curr + 1]) {
                    this.cellCtrl.trigger("swapRow", curr, curr + 1);
                    this.setGhostCover(curr + 1);
                    this.currentPosition++;
                } else if (typeof cps[curr - 1] === "number" && handleTop < cps[curr]) {
                    this.cellCtrl.trigger("swapRow", curr - 1, curr);
                    this.setGhostCover(curr - 1);
                    this.currentPosition--;
                }
            },

            onShow: function($cell) {
                if (this.cellCtrl.getRowCount() <= 1) 
                    return this.hide();

                this.cellHeight = this.cellCtrl.getCellHeight();
                if (this.$thead.find($cell).length === 0) {
                    this.$bar.css({
                        "top": $cell.position().top,
                        "height": this.cellCtrl.getCellHeight()
                    });
                } else {
                    this.hide();
                }
            }
        }),
        ColTitlebar = CellTitlebar.create({
            onInit: function() {
                this.$bar.addClass("bldr-table-titlecol");
                this.cssAxis = "left";
                this.$move.attr("title", translate('webs.module.table.moveColumn'));
                this.$delete.attr("title", translate('webs.module.table.deleteColumn'));
            },

            deleteClick: function() {
                this.cellCtrl.deleteCurrentCol();
            },

            getGhostContent: function() {
                var $cells = this.cellCtrl.getColCells(),
                content = '<thead class="' + $cells.parents("thead").eq(0).attr("class") + '">';
                $cells.each(function(i) {
                    var $cell = $cells.eq(i);
                    content += '<tr class="' + $cell.parent().attr("class") + '">' + 
                    '<td style="width: ' + $cell.width() + 'px;">' + $cell.html() + '</td></tr>';
                    if (i === 0) 
                        content += "</thead><tbody>";
                });
                content += "</tbody>";
                return content;
            },

            getAdjacentCells: function() {
                return this.cellCtrl.getRowCells();
            },
            getPosition: function() {
                return this.cellCtrl.getColPosition();
            },

            getGhostWidth: function() {
                return this.cellCtrl.getColWidth();
            },
            getGhostHeight: function() {
                return this.cellCtrl.getTableHeight();
            },
            getCellPosition: function(i) {
                return this.cellCtrl.getColCells(i).eq(1).position().left;
            },

            onDragStart: function(e) {
                this.el.addClass("moving-col");
            },

            moveGhost: function(e) {
                var handleLeft = e.clientX - this.elOffset.left - this.relOffset.x + 34;
                var left = handleLeft - this.cellWidth / 2;
                var top = e.clientY - this.elOffset.top - this.relOffset.y;
                var cps = this.cellPositions, curr = this.currentPosition;

                this.$bar.css({
                    "top": top,
                    "left": left
                });

                if (cps[curr + 1] && handleLeft > cps[curr + 1]) {
                    this.cellCtrl.trigger("swapCol", curr, curr + 1);
                    this.setGhostCover(curr + 1);
                    this.currentPosition++;
                } else if (handleLeft < cps[curr]) {
                    this.cellCtrl.trigger("swapCol", curr - 1, curr);
                    this.setGhostCover(curr - 1);
                    this.currentPosition--;
                }
            },

            onShow: function($cell) {
                if (this.cellCtrl.getColCount() <= 1) 
                    return this.hide();

                this.cellWidth = this.cellCtrl.getColWidth();
                this.$bar.css({
                    "left": $cell.position().left,
                    "width": this.cellWidth
                });
            }
        });



        /**
         * Controller for resizing column
         */
        var CellResizer = Spine.Controller.create({
            init: function() {
                this.$body = this.el.parents("body")
                .mousemove($.proxy(this, "mousemove"))
                .mouseup($.proxy(this, "mouseup"));
                this.$handle = $('<div class="bldr-resize-handle"/>').mousedown($.proxy(this, "mousedown"));
                this.$display = $('<div class="bldr-table-resize-display"/>');
                this.$bar = $('<div class="bldr-resize-column"/>').appendTo(this.el).append(this.$handle, this.$display);

                this.cellCtrl.bind("hide", $.proxy(this, "hide"));
                this.cellCtrl.bind("show", $.proxy(this, "show"));
            },

            mousedown: function(e) {
                this.moveflag = true;
                this.dragStart(e);
            },
            mousemove: function(e) {
                if (this.moveflag) {
                    this.dragResize(e);
                    return false;
                }
            },
            mouseup: function(e) {
                if (this.moveflag) {
                    this.moveflag = false;
                    this.dragEnd(e);
                }
            },

            getDragCell: function() {
                return this.cellCtrl.$table.find("tbody tr:nth-child(1) > td:nth-child(" + (this.colPos + 1) + ")");
            },

            dragStart: function(e) {
                this.el.addClass("resizing");
                this.$bar.addClass("resizing");
                this.$body.addClass("noselect");

                this.setColumnPosition(this.cellCtrl.getColPosition());
                this.$dragCell = this.getDragCell();
                this.$dragCellNext = this.$dragCell.next();

                this.leftCellStartWidth = this.$dragCell.width();
                this.totalCellsWidth = this.leftCellStartWidth + this.$dragCellNext.width();
                this.startClientX = e.clientX;

                this.tableCellsWidth = this.cellCtrl.getTableCellsWidth();

                this.minWidthLeft = null;
                this.minWidthRight = null;
            },

            dragResize: function(e) {
                var positionDiff = e.clientX - this.startClientX;
                var leftCellWidth = parseInt(this.leftCellStartWidth + positionDiff, 10);
                var rightCellWidth = this.totalCellsWidth - leftCellWidth, actualWidth;

                // If min-widths exist and we've hit it, normalize width
                if (this.minWidthLeft && leftCellWidth <= this.minWidthLeft) {
                    leftCellWidth = this.minWidthLeft;
                    rightCellWidth = this.totalCellsWidth - leftCellWidth;
                } else if (this.minWidthRight && rightCellWidth <= this.minWidthRight) {
                    rightCellWidth = this.minWidthRight;
                    leftCellWidth = this.totalCellsWidth - rightCellWidth;
                }

                // Set width and let the browser decide if we hit limits
                if (positionDiff < 0) {
                    this.$dragCell.width(leftCellWidth);
                    this.$dragCellNext.width("");
                } else {
                    this.$dragCell.width("");
                    this.$dragCellNext.width(rightCellWidth);
                }

                // min-width isn't set, grab the browser limit, then set again
                if (positionDiff < 0) {
                    // calculating left cell's min width
                    actualWidth = this.$dragCell.width();
                    if (leftCellWidth < actualWidth) 
                        this.minWidthLeft = leftCellWidth = actualWidth;
                    rightCellWidth = this.totalCellsWidth - leftCellWidth;
                } else {
                    // calculating right cell's min width
                    actualWidth = this.$dragCellNext.width();
                    if (rightCellWidth < actualWidth) 
                        this.minWidthRight = rightCellWidth = actualWidth;
                    leftCellWidth = this.totalCellsWidth - rightCellWidth;
                }

                this.$dragCell.width(leftCellWidth);
                this.$dragCellNext.width(rightCellWidth);

                this.setWidth(leftCellWidth, rightCellWidth);

                this.setPosition();
                this.cellCtrl.heightChanged();
            },

            dragEnd: function(e) {
                this.el.removeClass("resizing");
                this.$bar.removeClass("resizing");
                this.$body.removeClass("noselect");
                this.$activeCell.prev().mouseover();
            },

            /**
            	 * displays width on label, triggers a data setWidth with percentage Value
            	 */
            setWidth: function(w1) {
                this.$display.html(w1);
                this.cellCtrl.trigger("setWidth", this.colPos, w1 * 100 / this.tableCellsWidth);
            },

            show: function($cell) {
                if (!this.moveflag) {
                    this.$activeCell = $cell;
                    if (this.onShow) 
                        this.onShow();
                }
            },

            setPosition: function() {
                var left = this.$activeCell.position().left + this.getOffset();
                this.$bar.addClass("active").css("left", left);
            },

            hide: function() {
                this.$bar.removeClass("active");
            },
            getOffset: function() {
                return - 1;
            }
        }),
        LeftCellResizer = CellResizer.create({
            setColumnPosition: function(i) {
                this.colPos = i - 1;
            },
            onShow: function() {
                this.setPosition();
                if (this.cellCtrl.isFirstColumn()) 
                    this.hide();
            }
        }),
        RightCellResizer = CellResizer.create({
            setColumnPosition: function(i) {
                this.colPos = i;
            },
            onShow: function($cell) {
                if (this.cellCtrl.isLastColumn()) 
                    return this.hide();

                this.$activeCell = this.$activeCell.next();
                this.setPosition();
                if (!this.$activeCell[0]) 
                    return this.hide();
            }
        });


    })(mod, ext);

    return ModuleClassLoader.register('table', mod, ext);
});




// seo_footer Module (custom)

define('module/seo_footer', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.seo_footer'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return false;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return false;
        };



        // Module HTML Template
        extend.template = '<div class="webs-seo-footer" id="seo-footer" data-placeholder="Your seo-footer.">\
        <div class="site-info">\
        <h3 class="webs-title site-title">{{{title}}}</h3>\
        {{#if subtitleOn}}\
        <h4 class="webs-text site-subtitle">{{{subtitle}}}</h4>\
        {{/if}}\
        {{#if address}}\
        <address class="webs-text">\
        {{address.street1}}\
        {{#if address.street2}}\
        , {{address.street2}}\
        {{/if}}\
        {{#if address.city}}\
        , {{address.city}}\
        {{/if}}\
        {{address.state}}\
        {{#if address.city}}\
        {{address.country}}\
        {{else}}\
        {{#if address.state}}\
        {{address.country}}\
        {{/if}}\
        {{/if}}\
        {{address.postalCode}}\
        </address>\
        {{/if}}\
        <h6 class="webs-text copyright">&copy; {{year}} {{{title}}}. {{#l}}webs.module.seo_footer.allRightsReserved{{/l}}</h6>\
        </div>\
        <div class="seo-links webs-text">\
        {{#if links}}\
        {{#seoFooterLinks links}}\
        <a href="{{url}}" title="{{title}}">{{title}}</a>\
        {{/seoFooterLinks}}\
        {{/if}}\
        </div>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "title": "",
            "subtitle": "",
            "links": null,
            "address": null,
            "year": 2014
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS


        // Edit JS
        function stripHTML(string) {
            return string.replace(/(<([^>]+)>)/ig, "");
        }

        /**
        * Initialize edit mode
        */
        module.edit = function(event) {};

        module.initTooltip = function() {
            var self = this;
            try {
                this.$tt = bldr.pageController.$.tooltip({
                    content: "<span class='seo_footer-tooltip'>" + translate("webs.module.seo_footer.tooltip") + "</span>"
                });
                this.$tt.css('width', '280px');
                this.el.bind("mouseover", function() {
                    if (!bldr.pageController.dom.$body.find(".webs-bin-active").length)
                        self.$tt.addClass("active");
                });
                this.el.bind("mouseout", function() {
                    self.$tt.removeClass("active");
                });
            } catch (err) {
                webs.log.error("toolip init error in header editor", err);
            }
        };

        module.onDeactivate = function() {};

        module.safeOneLoaded = function() {
            var self = this;
            this.initTooltip();
            this.bind("rendered", function() {
                self.el.find("a").bind("click", function(e) {
                    e.stopPropagation();
                    e.preventDefault();
                    e.stopImmediatePropagation();
                });
            });

            var thisYear = new Date().getFullYear();

            if (self.data.year != thisYear) {
                self.data.year = thisYear;
                self.render();

                // Event listeners aren't set up yet, defer the call using setTimeout().
                setTimeout(function() {
                    self.dirty();
                }, 1);
            }

            // Keep title and subtitle synced with header editor
            Spine.Events.bind("headereditor:title:update", function(title) {
                self.data.title = stripHTML(title);
                self.render();
                self.dirty();
            });
            Spine.Events.bind("headereditor:subtitle:update", function(subtitle) {
                self.data.subtitle = stripHTML(subtitle);
                self.render();
                self.dirty();
            });
            Spine.Events.bind("headereditor:subtitle:toggle", function(on) {
                self.data.subtitleOn = on;
                self.render();
                self.dirty();
            });
        };


    })(mod, ext);

    return ModuleClassLoader.register('seo_footer', mod, ext);
});




// mobile_hours Module (custom)

define('module/mobile_hours', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.mobile_hours'],
function($, ModuleClassLoader, translate) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return false;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return false;
        };



        // Module HTML Template
        extend.template = '<div class="webs-mobile webs-mobile-hours" id="mobile_hours">\
        <ul>\
        {{#each days}}\
        {{#unless hidden}} \
        <li class="{{id}}">\
        <span class="day">{{#l}}{{transKey}}{{/l}}</span>\
        {{#if open}}\
        <span class="hours">\
        {{formatTime hours.start.h hours.start.mm hours.start.ampm}} - {{formatTime hours.stop.h hours.stop.mm hours.stop.ampm}}\
        </span>\
        {{else}}\
        {{#l}}webs.module.mobile_hours.closed{{/l}}\
        {{/if}}\
        </li>\
        {{/unless}}\
        {{/each}}\
        </ul>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "days": [{
                "id": "Monday",
                "transKey": "webs.module.mobile_hours.day.Monday",
                "hours": {
                    "start": {
                        "h": 8,
                        "mm": 0,
                        "ampm": "am"
                    },
                    "stop": {
                        "h": 9,
                        "mm": 0,
                        "ampm": "pm"
                    }
                },
                "hidden": false,
                "open": true
            }, {
                "id": "Tuesday",
                "transKey": "webs.module.mobile_hours.day.Tuesday",
                "hours": {
                    "start": {
                        "h": 8,
                        "mm": 0,
                        "ampm": "am"
                    },
                    "stop": {
                        "h": 9,
                        "mm": 0,
                        "ampm": "pm"
                    }
                },
                "hidden": false,
                "open": true
            }, {
                "id": "Wednesday",
                "transKey": "webs.module.mobile_hours.day.Wednesday",
                "hours": {
                    "start": {
                        "h": 8,
                        "mm": 0,
                        "ampm": "am"
                    },
                    "stop": {
                        "h": 9,
                        "mm": 0,
                        "ampm": "pm"
                    }
                },
                "hidden": false,
                "open": true
            }, {
                "id": "Thursday",
                "transKey": "webs.module.mobile_hours.day.Thursday",
                "hours": {
                    "start": {
                        "h": 8,
                        "mm": 0,
                        "ampm": "am"
                    },
                    "stop": {
                        "h": 9,
                        "mm": 0,
                        "ampm": "pm"
                    }
                },
                "hidden": false,
                "open": true
            }, {
                "id": "Friday",
                "transKey": "webs.module.mobile_hours.day.Friday",
                "hours": {
                    "start": {
                        "h": 8,
                        "mm": 0,
                        "ampm": "am"
                    },
                    "stop": {
                        "h": 9,
                        "mm": 0,
                        "ampm": "pm"
                    }
                },
                "hidden": false,
                "open": true
            }, {
                "id": "Saturday",
                "transKey": "webs.module.mobile_hours.day.Saturday",
                "hours": {
                    "start": {
                        "h": 8,
                        "mm": 0,
                        "ampm": "am"
                    },
                    "stop": {
                        "h": 9,
                        "mm": 0,
                        "ampm": "pm"
                    }
                },
                "hidden": false,
                "open": true
            }, {
                "id": "Sunday",
                "transKey": "webs.module.mobile_hours.day.Sunday",
                "hidden": false,
                "open": false
            }
            ]
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS
        module.oneLoaded = function() {
            var self = this,
            days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
            day = days[(new Date()).getDay()];
            this.el.find('li').hide().filter('.' + day).show();
            this.el.find('.webs-mobile-hours').addClass('webs-mobile-hours-today');

            this.el.click(function() {
                self.el.find('li').show();
                self.el.find('.webs-mobile-hours').removeClass('webs-mobile-hours-today');
            });
        };


        // Edit JS
        /**
        * Initialize edit mode
        */
        module.edit = function(event) {};

        module.initTooltip = function() {};

        module.onDeactivate = function() {};

        module.safeOneLoaded = function() {};


    })(mod, ext);

    return ModuleClassLoader.register('mobile_hours', mod, ext);
});




// mobile_email Module (custom)

define('module/mobile_email', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader'],
function($, ModuleClassLoader) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return false;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return false;
        };



        // Module HTML Template
        extend.template = '<a class="webs-mobile webs-mobile-email" href="mailto:{{emailAddress}}">{{emailAddress}}</a>\
        ';

        // Module Default Data
        extend.defaultData = {
            "emailAddress": "place@holder.com"
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "default": {
                "global": {
                    "css": "view.less"
                },
                "slug": "default"
            }
        };
        if (!extend.styles['default']['global']) {
            extend.styles['default']['global'] = {};
        }
        extend.styles['default']['global']['js'] = null;
        extend.defaultStyle = extend.styles['default'];

        // Register Toolbars

        // View JS


        // Edit JS
        /**
        * Initialize edit mode
        */
        module.edit = function(event) {};

        module.initTooltip = function() {};

        module.onDeactivate = function() {};

        module.safeOneLoaded = function() {};


    })(mod, ext);

    return ModuleClassLoader.register('mobile_email', mod, ext);
});




// photo_gallery Module (custom)

define('module/photo_gallery', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.photo_gallery', 'underscore', 'backbone', 'site/deckInspectah/DeckInspectahView', 'site/modules/photo_gallery/PhotoGalleryDeckInspectahImageView', 'site/modules/photo_gallery/PhotoGalleryDeckInspectahSidebarView', 'site/modules/photo_gallery/PhotoGalleryPhoto', 'site/modules/photo_gallery/SwappableSortableView', 'site/modules/photo_gallery/PhotoGalleryImageView'],
function($, ModuleClassLoader, translate, _, Backbone, DeckInspectahView, PhotoGalleryDeckInspectahImageView, PhotoGalleryDeckInspectahSidebarView, PhotoGalleryPhoto, SwappableSortableView, PhotoGalleryImageView) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="moveable"/>\
        {{#case viewType "grid"}}\
        <div class="w-photo_gallery grid {{settings.grid.gridPadding}} {{settings.grid.cropStyle}}">\
        <div class="row">\
        {{#each images}}\
        <div class="cell" data-_i_="{{_i_}}">\
        <div class="cell-wrapper">\
        <div class="photo-settings">\
        <a class="crop_photo" title="{{#l}}webs.module.photo_gallery.cropPhoto{{/l}}">\
        <img src="http://static.websimages.com/static/projects/finch/images/photo_gallery-crop.png">\
        </a>\
        <a class="edit_caption" title="{{#l}}webs.module.photo_gallery.dialog.editCaption{{/l}}">\
        <img src="http://static.websimages.com/static/projects/finch/images/photo_gallery-comment.png">\
        </a>\
        <a class="remove_photo last" title="{{#l}}webs.module.photo_gallery.dialog.removePhoto{{/l}}">\
        <img src="http://static.websimages.com/static/projects/finch/images/photo_gallery-delete.png">\
        </a>\
        </div>\
        \
        <div class="arrow">\
        </div>\
        <div class="titlecaption-container">\
        <div class="inner-titlecaption-container">\
        <input class="title" type="text"\
        placeholder="{{#l}}webs.module.photo_gallery.dialog.setting.titlePlaceholder{{/l}}"\
        value="{{this.title}}"/>\
        <textarea class="caption"\
        placeholder="{{#l}}webs.module.photo_gallery.dialog.setting.captionPlaceholder{{/l}}">{{this.caption}}</textarea>\
        </div>\
        </div>\
        \
        <div class="image-container"></div>\
        {{#buildTitleCaption ../settings.grid this}}\
        <div class="webs-gallery-grid-title-caption-social-box w-font-lato">\
        {{#if showTitle}}\
        <div class="webs-gallery-grid-title ellip-container">\
        <div class="ellip-wrap">\
        <p class="ellip-content">{{#if this.title}}{{this.title}}{{else}}\
        &nbsp;{{/if}}</p>\
        </div>\
        <span class="ellipsis">\
        <a href="#" title="{{#l}}webs.module.photo_gallery.showLightbox{{/l}}"\
        class="ellip-anchor">&hellip;</a>\
        </span>\
        </div>\
        {{/if}}\
        {{#if showCaption}}\
        <div class="webs-gallery-grid-caption ellip-container">\
        <div class="ellip-wrap">\
        <p class="ellip-content">{{#if this.caption}}{{this.caption}}{{else}}\
        &nbsp;{{/if}}</p>\
        </div>\
        <span class="ellipsis">\
        <a href="#" title="{{#l}}webs.module.photo_gallery.showLightbox{{/l}}"\
        class="ellip-anchor">&hellip;</a>\
        </span>\
        </div>\
        {{/if}}\
        {{#if showshowSocial}}\
        <div class="webs-gallery-grid-social"><!-- Not enabled--></div>{{/if}}\
        </div>\
        {{/buildTitleCaption}}\
        </div>\
        </div>\
        {{{_every_ ../settings.grid.photosPerRow "</div><div class=\'row\'>"}}}\
        {{/each}}\
        {{#unless images.length}}\
        <div class="bldr-placeholder-photo-gallery bldr-placeholder-element">\
        <span class="placeholder-text">{{#l}}webs.module.photo_gallery.view.noImage{{/l}}</span>\
        </div>\
        {{/unless}}\
        </div>\
        </div>\
        {{/case}}\
        {{#case viewType "slideshow"}}\
        <div class="w-photo_gallery slideshow {{settings.slideshow.cropStyle}} {{settings.slideshow.thumbnailPosition}}">\
        {{#if images.length}}\
        <div class="slide-container {{settings.slideshow.thumbnailPosition}}">\
        \
        <div class="slide-container-1" style="background-color:{{../settings.slideshow.backgroundColor}}">\
        {{#if displaySlideButton}}\
        <div class="button left-button {{settings.slideshow.buttonStyle}}"></div>\
        <div class="button right-button {{settings.slideshow.buttonStyle}}"></div>\
        {{/if}}\
        {{#each images}}\
        <div class="slide" data-_i_="{{_i_}}">\
        <div class="wrap-container">\
        <div class="slide-wrapper {{../settings.slideshow.transition}} {{#if selected}}slide-selected{{/if}}\
        "data-_i_="{{_i_}}">\
        <div class="photo-settings">\
        <a class="crop_photo" title="{{#l}}webs.module.photo_gallery.cropPhoto{{/l}}">\
        <img src="http://static.websimages.com/static/projects/finch/images/photo_gallery-crop.png">\
        </a>\
        <a class="edit_caption"\
        title="{{#l}}webs.module.photo_gallery.dialog.editCaption{{/l}}">\
        <img src="http://static.websimages.com/static/projects/finch/images/photo_gallery-comment.png">\
        </a>\
        <a class="remove_photo last"\
        title="{{#l}}webs.module.photo_gallery.dialog.removePhoto{{/l}}">\
        <img src="http://static.websimages.com/static/projects/finch/images/photo_gallery-delete.png">\
        </a>\
        </div>\
        \
        <div class="arrow"></div>\
        <div class="image-container"></div>\
        <div class="titlecaption-container">\
        <div class="inner-titlecaption-container">\
        <input class="title" type="text" placeholder="{{#l}}webs.module.photo_gallery.dialog.setting.titlePlaceholder{{/l}}" value="{{this.title}}"/>\
        <textarea class="caption" placeholder="{{#l}}webs.module.photo_gallery.dialog.setting.captionPlaceholder{{/l}}">{{this.caption}}</textarea>\
        </div>\
        </div>\
        </div>\
        </div>\
        </div>\
        {{/each}}\
        </div>\
        \
        <div class="title-wrapper {{settings.slideshow.showTitle}} {{settings.slideshow.showCaption}}">\
        {{#each images}}\
        {{#buildTitleCaption ../settings.slideshow this}}\
        <div class="webs-gallery-grid-title-caption-social-box w-font-lato" data-_i_="{{_i_}}">\
        {{#if showTitle}}\
        <div class="webs-gallery-grid-title ellip-container">\
        <div class="ellip-wrap">\
        <p class="ellip-content">{{#if this.title}}{{this.title}}{{else}}&nbsp;{{/if}}</p>\
        </div>\
        <span class="ellipsis">\
        <a href="#" title="{{#l}}webs.module.photo_gallery.showLightbox{{/l}}"\
        class="ellip-anchor">&hellip;</a>\
        </span>\
        </div>\
        {{/if}}\
        {{#if showCaption}}\
        <div class="webs-gallery-grid-caption ellip-container">\
        <div class="ellip-wrap">\
        <p class="ellip-content">{{#if this.caption}}{{this.caption}}{{else}}&nbsp;{{/if}}</p>\
        </div>\
        <span class="ellipsis">\
        <a href="#" title="{{#l}}webs.module.photo_gallery.showLightbox{{/l}}"\
        class="ellip-anchor">&hellip;</a>\
        </span>\
        </div>\
        {{/if}}\
        {{#if showshowSocial}}\
        <div class="webs-gallery-grid-social"><!-- Not enabled--></div>{{/if}}\
        </div>\
        {{/buildTitleCaption}}\
        {{/each}}\
        </div>\
        </div>\
        <div class="w-slideshow-thumbnail-area {{settings.slideshow.thumbnailPosition}}">\
        <div class="w-slideshow-thumbnails">\
        <div class="w-slideshow-thumbnails-container">\
        <ul class="w-slideshow-thumbnails-ul {{settings.slideshow.thumbnailPosition}}">\
        {{#each images}}\
        <li class="thumbnail {{../settings.slideshow.thumbnailPosition}} {{#if selected}}slide-selected{{/if}}"\
        data-_i_="{{_i_}}">\
        <figure class="w-slideshow-figure">\
        <div class="w-slideshow-image_container clearfix">\
        <img src="{{manipulate_image . square="250"}}" alt="{{caption}}"\
        class="w-slideshow-image">\
        {{#if caption}}\
        <figcaption class="w-slideshow-caption">\
        <div>{{caption}}</div>\
        </figcaption>{{/if}}\
        </div>\
        </figure>\
        </li>\
        {{/each}}\
        </ul>\
        </div>\
        {{#if displayThumbnailButton}}\
        <div class="w-slideshow-thumbnails-nav">\
        <div class="w-slideshow-thumbnail-button w-slideshow-thumbnail-button-prev {{../settings.slideshow.thumbnailPosition}}"></div>\
        <div class="w-slideshow-thumbnail-button w-slideshow-thumbnail-button-next {{../settings.slideshow.thumbnailPosition}}"></div>\
        </div>\
        {{/if}}\
        </div>\
        </div>\
        {{/if}}\
        \
        {{#unless images.length}}\
        <div class="bldr-placeholder-photo-gallery bldr-placeholder-element">\
        <span class="placeholder-text">{{#l}}webs.module.photo_gallery.view.noImage{{/l}}</span>\
        </div>\
        {{/unless}}\
        \
        </div>\
        {{/case}}\
        \
        <div class="placeholder-template" style="display: none">\
        <div class="cell">\
        <div class="cell-wrapper">\
        <div class="image-edit-wrapper">\
        <div class="webs-image-wrapper-1">\
        <div class="webs-image-crop">\
        <div class="placeholderText">\
        {{#l}}webs.module.photo_gallery.placeholder.text{{/l}}\
        </div>\
        </div>\
        </div>\
        </div>\
        </div>\
        </div>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "viewType": "grid",
            "showCropExplanation": true,
            "images": [],
            "settings": {
                "grid": {
                    "cropStyle": "crop-style-square",
                    "showTitle": "never",
                    "showCaption": "never",
                    "photosPerRow": 3,
                    "gridPadding": "padding-medium",
                    "deckInspectah": true,
                    "hoverEffect": "hover-effect-zoomSign",
                    "hoverEffectColor": "hover-effect-color-black"
                },
                "slideshow": {
                    "cropStyle": "crop-style-widescreen",
                    "showTitle": "never",
                    "showCaption": "never",
                    "duration": 3000,
                    "buttonStyle": "style1",
                    "transition": "scale",
                    "thumbnailPosition": "thumbnail-horizontal thumbnail-bottom",
                    "backgroundColor": "#222222"
                }
            }
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "base": {
                "global": {
                    "css": "view.less"
                },
                "slug": "base"
            }
        };
        if (!extend.styles['base']['global']) {
            extend.styles['base']['global'] = {};
        }
        extend.styles['base']['global']['js'] = null;
        extend.defaultStyle = extend.styles['base'];

        // Register Toolbars
        bldr.toolbar.register('photo_gallery', {
            "icon": "photo_gallery",
            "items": [{
                "label": "photo_gallery.manageImages",
                "name": "images",
                "dialog": {
                    "url": "/s/modules/photo_gallery/v1.66/photo_gallery.dialog.html",
                    "heading": "Gallery Settings",
                    "localizeHeading": "webs.module.photo_gallery.dialog.header",
                    "width": 750,
                    "height": 500
                },
                "style": "blue"
            }, {
                "label": "photo_gallery.viewType",
                "menuLabel": "photo_gallery.viewType",
                "type": "dropdown",
                "name": "viewType",
                "width": 100,
                "display": "inline",
                "options": [{
                    "label": "photo_gallery.Grid",
                    "value": "grid"
                }, {
                    "label": "photo_gallery.Slideshow",
                    "value": "slideshow"
                }
                ]
            }, {
                "type": "menu",
                "label": "photo_gallery.gallerySettings",
                "menuLabel": "Gallery Settings",
                "menu": [{
                    "type": "dropdown",
                    "label": "photo_gallery.photosPerRow",
                    "name": "photosPerRow",
                    "width": 50,
                    "display": "inline",
                    "options": [{
                        "label": 1,
                        "value": 1
                    }, {
                        "label": 2,
                        "value": 2
                    }, {
                        "label": 3,
                        "value": 3
                    }, {
                        "label": 4,
                        "value": 4
                    }, {
                        "label": 5,
                        "value": 5
                    }, {
                        "label": 6,
                        "value": 6
                    }
                    ]
                }, {
                    "type": "dropdown",
                    "label": "photo_gallery.cropStyle",
                    "name": "cropStyle",
                    "width": 95,
                    "display": "inline",
                    "options": [{
                        "label": "photo_gallery.cropStyleSquare",
                        "value": "crop-style-square"
                    }, {
                        "label": "photo_gallery.cropStyleRectangle",
                        "value": "crop-style-rectangle"
                    }, {
                        "label": "photo_gallery.cropStylePortrait",
                        "value": "crop-style-portrait"
                    }, {
                        "label": "photo_gallery_slideshow.cropStyleWidescreen",
                        "value": "crop-style-widescreen"
                    }
                    ]
                }, {
                    "type": "dropdown",
                    "label": "photo_gallery.gridPadding",
                    "name": "gridPadding",
                    "width": 80,
                    "display": "inline",
                    "options": [{
                        "label": "photo_gallery.paddingLarge",
                        "value": "padding-large"
                    }, {
                        "label": "photo_gallery.paddingMedium",
                        "value": "padding-medium"
                    }, {
                        "label": "photo_gallery.paddingSmall",
                        "value": "padding-small"
                    }, {
                        "label": "photo_gallery.paddingNone",
                        "value": "padding-none"
                    }
                    ]
                }, {
                    "display": "inline",
                    "label": "photo_gallery.lightbox",
                    "type": "switch-bare",
                    "name": "deckInspectah",
                    "tooltip": "photo_gallery.lightboxTooltip"
                }, {
                    "display": "inline",
                    "type": "dropdown",
                    "label": "photo_gallery.hoverEffect",
                    "name": "hoverEffect",
                    "width": 80,
                    "options": [{
                        "label": "photo_gallery.hoverEffect.none",
                        "value": "hover-effect-none"
                    }, {
                        "label": "photo_gallery.hoverEffect.zoomSign",
                        "value": "hover-effect-zoomSign"
                    }, {
                        "label": "photo_gallery.hoverEffect.plusSign",
                        "value": "hover-effect-plusSign"
                    }, {
                        "label": "photo_gallery.hoverEffect.viewSign",
                        "value": "hover-effect-viewSign"
                    }, {
                        "label": "photo_gallery.hoverEffect.titleSign",
                        "value": "hover-effect-titleSign"
                    }
                    ]
                }, {
                    "display": "inline",
                    "type": "dropdown",
                    "label": "photo_gallery.color",
                    "name": "hoverEffectColor",
                    "width": 90,
                    "options": [{
                        "label": "photo_gallery.hoverEffectColor.black",
                        "value": "hover-effect-color-black"
                    }, {
                        "label": "photo_gallery.hoverEffectColor.white",
                        "value": "hover-effect-color-white"
                    }
                    ]
                }, {
                    "type": "dropdown",
                    "label": "photo_gallery.title",
                    "name": "showTitle",
                    "width": 250,
                    "display": "inline",
                    "options": [{
                        "label": "photo_gallery.titlecaption.lightboxOnly",
                        "value": "lightbox"
                    }, {
                        "label": "photo_gallery.titlecaption.gridOnly",
                        "value": "grid"
                    }, {
                        "label": "photo_gallery.titlecaption.bothViews",
                        "value": "both"
                    }, {
                        "label": "photo_gallery.titlecaption.never",
                        "value": "never"
                    }
                    ]
                }, {
                    "type": "dropdown",
                    "label": "photo_gallery.caption",
                    "name": "showCaption",
                    "width": 250,
                    "display": "inline",
                    "options": [{
                        "label": "photo_gallery.titlecaption.lightboxOnly",
                        "value": "lightbox"
                    }, {
                        "label": "photo_gallery.titlecaption.gridOnly",
                        "value": "grid"
                    }, {
                        "label": "photo_gallery.titlecaption.bothViews",
                        "value": "both"
                    }, {
                        "label": "photo_gallery.titlecaption.never",
                        "value": "never"
                    }
                    ]
                }
                ]
            }
            ]
        });
        bldr.toolbar.register('photo_gallery-ramp', {
            "icon": "photo_gallery",
            "items": [{
                "label": "photo_gallery.manageImages",
                "name": "images",
                "dialog": {
                    "url": "/s/modules/photo_gallery/v1.66/photo_gallery.dialog.html",
                    "heading": "Gallery Settings",
                    "localizeHeading": "webs.module.photo_gallery.dialog.header",
                    "width": 750,
                    "height": 500
                },
                "style": "blue"
            }, {
                "type": "menu",
                "label": "photo_gallery.gallerySettings",
                "menuLabel": "Gallery Settings",
                "menu": [{
                    "type": "dropdown",
                    "label": "photo_gallery.photosPerRow",
                    "name": "photosPerRow",
                    "width": 50,
                    "display": "inline",
                    "options": [{
                        "label": 1,
                        "value": 1
                    }, {
                        "label": 2,
                        "value": 2
                    }, {
                        "label": 3,
                        "value": 3
                    }, {
                        "label": 4,
                        "value": 4
                    }, {
                        "label": 5,
                        "value": 5
                    }, {
                        "label": 6,
                        "value": 6
                    }
                    ]
                }, {
                    "type": "dropdown",
                    "label": "photo_gallery.cropStyle",
                    "name": "cropStyle",
                    "width": 95,
                    "display": "inline",
                    "options": [{
                        "label": "photo_gallery.cropStyleSquare",
                        "value": "crop-style-square"
                    }, {
                        "label": "photo_gallery.cropStyleRectangle",
                        "value": "crop-style-rectangle"
                    }, {
                        "label": "photo_gallery.cropStylePortrait",
                        "value": "crop-style-portrait"
                    }, {
                        "label": "photo_gallery_slideshow.cropStyleWidescreen",
                        "value": "crop-style-widescreen"
                    }
                    ]
                }, {
                    "type": "dropdown",
                    "label": "photo_gallery.gridPadding",
                    "name": "gridPadding",
                    "width": 80,
                    "display": "inline",
                    "options": [{
                        "label": "photo_gallery.paddingLarge",
                        "value": "padding-large"
                    }, {
                        "label": "photo_gallery.paddingMedium",
                        "value": "padding-medium"
                    }, {
                        "label": "photo_gallery.paddingSmall",
                        "value": "padding-small"
                    }, {
                        "label": "photo_gallery.paddingNone",
                        "value": "padding-none"
                    }
                    ]
                }, {
                    "display": "inline",
                    "label": "photo_gallery.lightbox",
                    "type": "switch-bare",
                    "name": "deckInspectah",
                    "tooltip": "photo_gallery.lightboxTooltip"
                }, {
                    "display": "inline",
                    "type": "dropdown",
                    "label": "photo_gallery.hoverEffect",
                    "name": "hoverEffect",
                    "width": 80,
                    "options": [{
                        "label": "photo_gallery.hoverEffect.none",
                        "value": "hover-effect-none"
                    }, {
                        "label": "photo_gallery.hoverEffect.zoomSign",
                        "value": "hover-effect-zoomSign"
                    }, {
                        "label": "photo_gallery.hoverEffect.plusSign",
                        "value": "hover-effect-plusSign"
                    }, {
                        "label": "photo_gallery.hoverEffect.viewSign",
                        "value": "hover-effect-viewSign"
                    }, {
                        "label": "photo_gallery.hoverEffect.titleSign",
                        "value": "hover-effect-titleSign"
                    }
                    ]
                }, {
                    "display": "inline",
                    "type": "dropdown",
                    "label": "photo_gallery.color",
                    "name": "hoverEffectColor",
                    "width": 90,
                    "options": [{
                        "label": "photo_gallery.hoverEffectColor.black",
                        "value": "hover-effect-color-black"
                    }, {
                        "label": "photo_gallery.hoverEffectColor.white",
                        "value": "hover-effect-color-white"
                    }
                    ]
                }, {
                    "type": "dropdown",
                    "label": "photo_gallery.title",
                    "name": "showTitle",
                    "width": 250,
                    "display": "inline",
                    "options": [{
                        "label": "photo_gallery.titlecaption.lightboxOnly",
                        "value": "lightbox"
                    }, {
                        "label": "photo_gallery.titlecaption.gridOnly",
                        "value": "grid"
                    }, {
                        "label": "photo_gallery.titlecaption.bothViews",
                        "value": "both"
                    }, {
                        "label": "photo_gallery.titlecaption.never",
                        "value": "never"
                    }
                    ]
                }, {
                    "type": "dropdown",
                    "label": "photo_gallery.caption",
                    "name": "showCaption",
                    "width": 250,
                    "display": "inline",
                    "options": [{
                        "label": "photo_gallery.titlecaption.lightboxOnly",
                        "value": "lightbox"
                    }, {
                        "label": "photo_gallery.titlecaption.gridOnly",
                        "value": "grid"
                    }, {
                        "label": "photo_gallery.titlecaption.bothViews",
                        "value": "both"
                    }, {
                        "label": "photo_gallery.titlecaption.never",
                        "value": "never"
                    }
                    ]
                }
                ]
            }
            ]
        });
        bldr.toolbar.register('photo_gallery-slideshow', {
            "icon": "photo_gallery",
            "items": [{
                "label": "photo_gallery.manageImages",
                "name": "images",
                "dialog": {
                    "url": "/s/modules/photo_gallery/v1.66/photo_gallery.dialog.html",
                    "heading": "Gallery Settings",
                    "localizeHeading": "webs.module.photo_gallery.dialog.header",
                    "width": 750,
                    "height": 500
                },
                "style": "blue"
            }, {
                "label": "photo_gallery.viewType",
                "menuLabel": "photo_gallery.viewType",
                "type": "dropdown",
                "name": "viewType",
                "width": 100,
                "display": "inline",
                "options": [{
                    "label": "photo_gallery.Grid",
                    "value": "grid"
                }, {
                    "label": "photo_gallery.Slideshow",
                    "value": "slideshow"
                }
                ]
            }, {
                "type": "menu",
                "label": "photo_gallery.slideshowSettings",
                "menuLabel": "Slideshow Settings",
                "menu": [{
                    "display": "inline",
                    "label": "Autoplay",
                    "type": "dropdown",
                    "name": "duration",
                    "width": 100,
                    "options": [{
                        "label": "Off",
                        "value": 0
                    }, {
                        "label": "1 Second",
                        "value": 1000
                    }, {
                        "label": "2 Seconds",
                        "value": 2000
                    }, {
                        "label": "3 Seconds",
                        "value": 3000
                    }, {
                        "label": "4 Seconds",
                        "value": 4000
                    }, {
                        "label": "5 Seconds",
                        "value": 5000
                    }
                    ]
                }, {
                    "display": "inline",
                    "label": "photo_gallery_slideshow.buttonStyle",
                    "type": "dropdown",
                    "name": "buttonStyle",
                    "width": 100,
                    "options": [{
                        "label": "photo_gallery_slideshow.buttonStyle1",
                        "value": "style1"
                    }, {
                        "label": "photo_gallery_slideshow.buttonStyle2",
                        "value": "style2"
                    }, {
                        "label": "photo_gallery_slideshow.buttonStyle3",
                        "value": "style3"
                    }, {
                        "label": "photo_gallery_slideshow.buttonStyle4",
                        "value": "style4"
                    }, {
                        "label": "photo_gallery_slideshow.buttonStyle5",
                        "value": "style5"
                    }, {
                        "label": "photo_gallery_slideshow.buttonStyle6",
                        "value": "style6"
                    }
                    ]
                }, {
                    "display": "inline",
                    "label": "photo_gallery.cropStyleSlideshow",
                    "type": "dropdown",
                    "name": "cropStyle",
                    "width": 100,
                    "options": [{
                        "label": "photo_gallery_slideshow.cropStyleSquare",
                        "value": "crop-style-square"
                    }, {
                        "label": "photo_gallery_slideshow.cropStyleRectangle",
                        "value": "crop-style-rectangle"
                    }, {
                        "label": "photo_gallery_slideshow.cropStyleWidescreen",
                        "value": "crop-style-widescreen"
                    }, {
                        "label": "photo_gallery_slideshow.cropStylePortrait",
                        "value": "crop-style-portrait"
                    }
                    ]
                }, {
                    "display": "inline",
                    "label": "thumbnailPosition",
                    "type": "dropdown",
                    "name": "thumbnailPosition",
                    "width": 100,
                    "options": [{
                        "label": "thumbnail-none",
                        "value": "thumbnail-none"
                    }, {
                        "label": "thumbnail-bottom",
                        "value": "thumbnail-horizontal thumbnail-bottom"
                    }, {
                        "label": "thumbnail-top",
                        "value": "thumbnail-horizontal thumbnail-top"
                    }, {
                        "label": "thumbnail-left",
                        "value": "thumbnail-vertical thumbnail-left"
                    }, {
                        "label": "thumbnail-right",
                        "value": "thumbnail-vertical thumbnail-right"
                    }
                    ]
                }, {
                    "display": "inline",
                    "label": "Transition",
                    "type": "dropdown",
                    "name": "transition",
                    "width": 100,
                    "options": [{
                        "label": "None",
                        "value": "none"
                    }, {
                        "label": "Fade",
                        "value": "fade"
                    }, {
                        "label": "H Carousel",
                        "value": "carouselHorizontal"
                    }, {
                        "label": "photo_gallery.transition.Vertical",
                        "value": "vertical"
                    }, {
                        "label": "photo_gallery.transition.Scale",
                        "value": "scale"
                    }
                    ]
                }, {
                    "display": "inline",
                    "type": "colorpicker-menu",
                    "label": "Color",
                    "name": "backgroundColor"
                }, {
                    "type": "dropdown",
                    "label": "photo_gallery.title",
                    "name": "showTitle",
                    "width": 250,
                    "display": "inline",
                    "options": [{
                        "label": "photo_gallery.titlecaption.lightboxOnly",
                        "value": "lightbox"
                    }, {
                        "label": "photo_gallery.titlecaption.gridOnly",
                        "value": "grid"
                    }, {
                        "label": "photo_gallery.titlecaption.bothViews",
                        "value": "both"
                    }, {
                        "label": "photo_gallery.titlecaption.never",
                        "value": "never"
                    }
                    ]
                }, {
                    "type": "dropdown",
                    "label": "photo_gallery.caption",
                    "name": "showCaption",
                    "width": 250,
                    "display": "inline",
                    "options": [{
                        "label": "photo_gallery.titlecaption.lightboxOnly",
                        "value": "lightbox"
                    }, {
                        "label": "photo_gallery.titlecaption.gridOnly",
                        "value": "grid"
                    }, {
                        "label": "photo_gallery.titlecaption.bothViews",
                        "value": "both"
                    }, {
                        "label": "photo_gallery.titlecaption.never",
                        "value": "never"
                    }
                    ]
                }
                ]
            }
            ]
        });

        // View JS
        $.extend(module, {

            VIEW_TYPE: {
                'GRID': 'grid',
                'SLIDESHOW': 'slideshow'
            },

            /**
            	 * Init function
            	 */

            oneLoaded: function () {
                var self = this;
                if (self.data.viewType == self.VIEW_TYPE.SLIDESHOW) {
                    self._oneLoadedSlideShow();
                } else {
                    self._oneLoadedGrid();
                }
            },

            _oneLoadedGrid: function () {
                var self = this;

                // if we haven't already included the deckInspectah css, do so now, and set a global variable so we don't try to include it again
                if (!webs.deckInspectah) {
                    try {
                        $("head").append('<link rel="stylesheet" type="text/css" href="' + webs.props.staticServer + '/active-static/target/site/deckInspectah/deckInspectah.css"/>');
                        webs.deckInspectah = true;
                    } catch (ex) {
                        webs.log.error("Unable to setup Deck Inspectah", this.data, ex);
                    }
                }

                this.el.on('click', '.ellip-anchor, .deck-inspectah-link', function (evt) {
                    evt.preventDefault();

                    var gridSettings = self.data.settings.grid;
                    var isEllipAnchor = $(evt.target).hasClass('ellip-anchor');

                    var closestCell = $(this).closest('.cell');
                    var isMobile = closestCell.hasClass('mobile-image');
                    var image = closestCell.data('_i_');
                    var titleEnabled = (!isMobile) && (isEllipAnchor || gridSettings.showTitle === 'lightbox' || gridSettings.showTitle == 'both');
                    var captionEnabled = (!isMobile) && (isEllipAnchor || gridSettings.showCaption === 'lightbox' || gridSettings.showCaption == 'both');

                    var deckInspectahView = new DeckInspectahView({
                        model: new Backbone.Collection(self.data.images),
                        selectedImage: image,
                        imageViewClass: PhotoGalleryDeckInspectahImageView,
                        sidebarViewClass: PhotoGalleryDeckInspectahSidebarView,
                        titleEnabled: titleEnabled,
                        captionEnabled: captionEnabled
                    });
                    $('body').append(deckInspectahView.el);
                    deckInspectahView.show();
                });

                var overlayEllipsis = _.bind(this._overlayEllipsis, this);
                try {
                    if (top && top.dsnr) {
                        top.require(['internal/sitebuilder/designerChrome/dsnrLoader'], function (dsnrLoader) {
                            dsnrLoader.sitePreview.bind('iframeRestyled', overlayEllipsis)
                        });
                    }
                } catch (e) {
                    // No designer - we can safely ignore
                }

                // always call the deferred, in case we are not on a designer frame
                _.defer(overlayEllipsis);
            },

            _oneLoadedSlideShow: function () {
                var self = this;
                var duration = self.data.settings.slideshow.duration;
                var autoplay = parseInt(duration, 10) != 0;
                if (autoplay && self.data.images.length > 1) {
                    this.autoplay(duration);
                }

                this.el.on('click', '.ellip-anchor, .deck-inspectah-link', function (evt) {
                    evt.preventDefault();

                    var gridSettings = self.data.settings.grid;
                    var isEllipAnchor = $(evt.target).hasClass('ellip-anchor');

                    var closestCell = $(this).closest('.slide');
                    var isMobile = closestCell.hasClass('mobile-image');
                    var image = closestCell.data('_i_');
                    var titleEnabled = (!isMobile);
                    var captionEnabled = (!isMobile);

                    var deckInspectahView = new DeckInspectahView({
                        model: new Backbone.Collection(self.data.images),
                        selectedImage: image,
                        imageViewClass: PhotoGalleryDeckInspectahImageView,
                        sidebarViewClass: PhotoGalleryDeckInspectahSidebarView,
                        titleEnabled: titleEnabled,
                        captionEnabled: captionEnabled
                    });
                    $('body').append(deckInspectahView.el);
                    deckInspectahView.show();
                });

                var overlayEllipsis = _.bind(this._overlayEllipsis, this);
                try {
                    if (top && top.dsnr) {
                        top.require(['internal/sitebuilder/designerChrome/dsnrLoader'], function (dsnrLoader) {
                            dsnrLoader.sitePreview.bind('iframeRestyled', overlayEllipsis)
                        });
                    }
                } catch (e) {
                    // No designer - we can safely ignore
                }

                // always call the deferred, in case we are not on a designer frame
                _.defer(overlayEllipsis);

                this.el.on('click', '.thumbnail', _.bind(self.thumbnailClick, self));
                this.el.on('mouseenter', '.w-slideshow-thumbnail-button', _.bind(self.thumbnailScroll, self));
                this.el.on('mouseleave', '.w-slideshow-thumbnail-button', _.bind(self.stopThumbnailScroll, self));

                //set the selected slide to the first one
                for (var i = 0; i < self.data.images.length; i++) {
                    self.el.find(".webs-gallery-grid-title-caption-social-box[data-_i_=" + i + "]").removeClass("slide-selected");
                    self.el.find(".slide-wrapper[data-_i_=" + i + "]").removeClass("slide-selected");
                    self.el.find(".thumbnail[data-_i_=" + i + "]").removeClass("slide-selected");
                    self.data.images[i].selected = false;
                }

                self.el.find(".webs-gallery-grid-title-caption-social-box[data-_i_=0]").addClass("slide-selected");
                self.el.find(".slide-wrapper[data-_i_=0]").addClass("slide-selected");
                self.el.find(".thumbnail[data-_i_=0]").addClass("slide-selected");
                self.data.images[0].selected = true;

                this.setupSlideshow();
            },

            _overlayEllipsis: function () {
                var self = this;
                // ignore elements that aren't showing title & caption
                if (this.$('.ellip-content')[0]) {
                    var parents = self.$('.ellip-content').eq(0).parents();
                    var bgColor = _.reduce(parents.toArray().reverse(), function (left, el) {
                        var rgba = $(el).css('background-color').match(/[\d\.]+/g);

                        // handle no background - webkit returns 0,0,0,0, gecko returns null
                        if (rgba && rgba.length < 4) {
                            //if there are only 1 (IE) or 3 (webkit/gecko) colors, then it's fully opaque
                            return rgba;
                        } else if (rgba && rgba[3] != 0) {
                            // ignore transparent backgrounds, otherwise do alpha blending
                            return _.map(rgba, function (val, key, rgba) {
                                return key === 3 ? 1 : Math.round(rgba[3] * val + (1 - rgba[3]) * left[key]);
                            });
                        } else {
                            // transparent, return the previous background
                            return left;
                        }
                    }, [255, 255, 255, 0]); // start with a white background

                    bgColor[3] = bgColor[3] || 1;

                    var rgbaStr = 'rgba(' + bgColor.toString() + ')';
                    var rgb = bgColor.slice(0, 3).toString(); // not creating rgbString as we need both rgba(x,x,x,0) and rgb(x,x,x)

                    this.$('.ellipsis').css('background-image',
                    'linear-gradient(to right, rgba(' + rgb + ',0), ' + rgbaStr + ' 50%, ' + rgbaStr + ')');

                    this.$('.ellip-anchor').css('background-color', 'rgb(' + rgb + ')');
                }
            },

            autoplay: function (duration) {
                var self = this;
                setInterval(function () {
                    self.nextSlide();
                }, duration);
            },

            setupSlideshow: function () {
                this.isMobile = this.el.find(".w-photo_gallery_slideshow-mobile").length;
                this.isEditMode = this.el.find(".moveable").length;
                this.data.currentSlide = 0;
                for (var i = 0; i < this.data.images.length; i++) {
                    if (this.data.images[i].selected) {
                        this.data.currentSlide = i;
                    }
                }

                var self = this;
                this.el.on('mousedown', '.right-button', function (e) {
                    self.nextSlide(e);
                });

                this.el.on('mousedown', '.left-button', function (e) {
                    self.prevSlide(e);
                });

                if (this.isMobile) {
                    this.mobileSetup(this);
                }

                this.duration = parseInt(this.data.settings.slideshow.duration, 10);
                if (this.duration && this.data.images.length > 1 && !this.isEditMode) {
                    this.isAutoplaying = true;
                    if (this.isMobile) {
                        this.$progress = this.el.find(".progress-timer");
                    }
                    this.playSlide();
                }
            },

            mobileSetup: function (context) {
                var self = this;
                this.setMobileSlideHeight(context);
                this.data.settings.slideshow.duration = 3000;

                //var $container = this.el.find(".slide-container");

                this.el.on("touchmove mousemove", function (e) {
                    e.stopPropagation();
                });

                this.el.on("dragstart", function (e) {
                    e.preventDefault();
                });

                this.el.on("mousedown", function (e) {
                    var ref = arguments.callee,
                    handle = $(this),
                    start = e.pageX;
                    handle.off("mousedown", ref);
                    handle.on("mouseup", function(e) {
                        handle.off("mouseup", arguments.callee);
                        handle.on("mousedown", ref);
                        var end = e.pageX;
                        if (Math.abs(end - start) > 30) {
                            handle.trigger((end > start) ? "swiperight" : "swipeleft");
                        };
                    });
                });

                this.el.on("swiperight", function() {
                    self.prevSlide();
                });

                this.el.on("swipeleft", function() {
                    self.nextSlide();
                });
            },

            setMobileSlideHeight: function (context) {
                var $container = this.el.find(".w-photo_gallery_slideshow-mobile").find(".slide-container"),
                $imgs = $container.find('img');
                _.each($imgs, function (img) {
                    var loading = new Image();
                    loading.onload = function() {
                        setTimeout(function() {
                            $container.css("height", loading.height);
                        }, 0);
                    };
                    loading.src = img.src;
                });
            },

            playSlide: function () {
                var self = this;
                if (this.isMobile) {
                    setTimeout(function () {
                        self.$progress.addClass("move");
                    }, 0);
                }
                this.timeout = setTimeout(function () {
                    if (self.isMobile) {
                        self.$progress.removeClass("move");
                    }
                    self.nextSlide();
                }, self.duration);
            },

            nextSlide: function () {
                if (this.data.images && this.data.images.length > 1) {
                    var prev = this.data.currentSlide;
                    var next = this.data.currentSlide + 1;
                    if (next == this.data.images.length) {
                        next = 0;
                    }
                    this.selectSlide(prev, next);
                }
            },

            prevSlide: function () {
                if (this.data.images && this.data.images.length > 1) {
                    var prev = this.data.currentSlide;
                    var next = this.data.currentSlide - 1;
                    if (next == - 1) {
                        next = this.data.images.length - 1;
                    }
                    this.selectSlide(prev, next);
                }
            },

            selectSlide: function (prev, next) {
                if (!this.thumbnailScrollIntervalRef) {
                    if (this.isAutoplaying) {
                        clearTimeout(this.timeout);
                        if (this.isMobile) {
                            this.$progress.removeClass("move");
                        }
                        this.playSlide();
                    }
                    var prevSlide = this.el.find(".slide-wrapper[data-_i_=" + prev + "]");
                    var nextSlide = this.el.find(".slide-wrapper[data-_i_=" + next + "]");
                    var transitionStyle = this.data.settings.slideshow.transition;
                    this.el.find(".slide-wrapper").removeClass("scalein scaleout incomingright outgoingleft incomingleft outgoingright");

                    if (transitionStyle == "carouselHorizontal" || transitionStyle == "vertical") {
                        if (next == 0 && prev == (this.data.images.length - 1)) {
                            nextSlide.addClass("incomingright");
                            prevSlide.addClass("outgoingleft");
                        } else if (prev == 0 && next == (this.data.images.length - 1)) {
                            nextSlide.addClass("incomingleft");
                            prevSlide.addClass("outgoingright");
                        } else if (prev < next) {
                            nextSlide.addClass("incomingright");
                            prevSlide.addClass("outgoingleft");
                        } else {
                            nextSlide.addClass("incomingleft");
                            prevSlide.addClass("outgoingright");
                        }
                    } else if (transitionStyle == "scale") {
                        nextSlide.addClass("scalein");
                        prevSlide.addClass("scaleout");
                    }

                    this.data.images[prev].selected = false;
                    prevSlide.removeClass("slide-selected");
                    nextSlide.addClass("slide-selected");
                    this.el.find(".webs-gallery-grid-title-caption-social-box[data-_i_=" + prev + "]").removeClass("slide-selected");
                    this.el.find(".webs-gallery-grid-title-caption-social-box[data-_i_=" + next + "]").addClass("slide-selected");
                    this.data.currentSlide = next;
                    this.data.images[next].selected = true;
                    if (!this.isMobile) {
                        this.adjustThumbnail(prev, next);
                    }
                }
            },

            adjustThumbnail: function (prev, next) {
                var self = this;

                var horizontalScroll = self.el.find('.w-slideshow-thumbnail-area').hasClass("thumbnail-horizontal");
                var thumbnailsContainer = self.el.find('.w-slideshow-thumbnails-ul');
                var thumbnailContainerWidth = thumbnailsContainer.height();
                var thumbnailsAreaWidth = self.el.find('.w-slideshow-thumbnails').height();
                if (horizontalScroll) {
                    thumbnailsAreaWidth = self.el.find('.w-slideshow-thumbnails').width();
                    thumbnailContainerWidth = thumbnailsContainer.width();
                }
                var oldThumbnail = self.el.find(".thumbnail[data-_i_=" + prev + "]");
                var newThumbnail = self.el.find(".thumbnail[data-_i_=" + next + "]");
                oldThumbnail.removeClass("slide-selected");
                newThumbnail.addClass("slide-selected");

                if (thumbnailContainerWidth > thumbnailsAreaWidth) {
                    var thumbnailWidth = parseInt(newThumbnail.height(), 10);
                    var left = parseInt(newThumbnail.position().top, 10);
                    if (horizontalScroll) {
                        left = parseInt(newThumbnail.position().left, 10);
                        thumbnailWidth = parseInt(newThumbnail.width(), 10);
                    }
                    var right = left + thumbnailWidth;
                    var jump = (((prev === 0) && (this.data.images.length - 1 === next)) || ((prev === this.data.images.length - 1) && (0 === next)));
                    if (!self.windowLeft) {
                        self.windowLeft = self.data.currentPosition;
                        self.windowRight = self.windowLeft + thumbnailsAreaWidth;
                    }
                }

                if ((right > self.windowRight) || (left < self.windowLeft)) {
                    if (right > self.windowRight) {
                        scrolling(true, 10, thumbnailContainerWidth - thumbnailsAreaWidth, thumbnailContainerWidth);
                    } else if (left < self.windowLeft) {
                        scrolling(false, - 10, 0, thumbnailsAreaWidth);
                    }
                }

                function scrolling(goRight, scrollingSpeed, winLef, winRig) {
                    self.thumbnailScrollIntervalRef = setInterval(function () {
                        thumbnailsContainer = self.el.find('.w-slideshow-thumbnails-ul');
                        var checkCondition = (right > self.windowRight);
                        if (!goRight) {
                            checkCondition = (left < self.windowLeft);
                        }
                        if (checkCondition) {
                            if (jump) {
                                // disable scroll for long jump
                                self.windowRight = winRig;
                                self.windowLeft = winLef;
                                self.data.currentPosition = - self.windowLeft;
                            } else {
                                self.windowLeft += scrollingSpeed;
                                self.windowRight += scrollingSpeed;
                                self.data.currentPosition -= scrollingSpeed;
                            }
                            if (horizontalScroll) {
                                thumbnailsContainer.css('left', self.data.currentPosition);
                            } else {
                                thumbnailsContainer.css('top', self.data.currentPosition);
                            }
                        } else {
                            clearInterval(self.thumbnailScrollIntervalRef);
                            self.thumbnailScrollIntervalRef = false;
                        }
                    }, 16);
                }
            },

            thumbnailClick: function (evt) {
                var targetDiv = $(evt.target);
                var prev = targetDiv.parents().siblings('.thumbnail.slide-selected').data('_i_');
                var next = targetDiv.parents('.thumbnail').data('_i_');
                if (typeof prev !== 'undefined' && prev != next) {
                    this.selectSlide(prev, next);
                }
            },

            thumbnailScroll: function (evt) {
                if (!this.thumbnailScrollIntervalRef) {
                    var self = this;
                    var targetDiv = $(evt.target);
                    var horizontalScroll = targetDiv.hasClass("thumbnail-horizontal");
                    var thumbnailsContainer = targetDiv.parents('.w-slideshow-thumbnails').find('.w-slideshow-thumbnails-ul');
                    var thumbnailsAreaWidth = targetDiv.parents('.w-slideshow-thumbnails').height();
                    var thumbnailContainerWidth = thumbnailsContainer.height();
                    if (horizontalScroll) {
                        thumbnailsAreaWidth = targetDiv.parents('.w-slideshow-thumbnails').width();
                        thumbnailContainerWidth = thumbnailsContainer.width();
                    }
                    if (thumbnailContainerWidth > thumbnailsAreaWidth) {
                        if (!self.windowLeft) {
                            self.windowLeft = self.data.currentPosition;
                            self.windowRight = self.windowLeft + thumbnailsAreaWidth;
                        }
                        if (targetDiv.hasClass('w-slideshow-thumbnail-button-next')) {
                            scrolling(true, 5);
                        } else if ((targetDiv.hasClass('w-slideshow-thumbnail-button-prev'))) {
                            scrolling(false, - 5);
                        }
                    }
                }

                function scrolling(next, scrollingSpeed) {
                    self.thumbnailScrollIntervalRef = setInterval(function () {
                        var checkCondition = (self.data.currentPosition < 0);
                        if (next) {
                            checkCondition = ((thumbnailContainerWidth - thumbnailsAreaWidth + self.data.currentPosition) > 0);
                        }
                        if (checkCondition) {
                            self.windowLeft += scrollingSpeed;
                            self.windowRight += scrollingSpeed;
                            self.data.currentPosition -= scrollingSpeed;
                            if (self.data.currentPosition > 0) {
                                self.data.currentPosition = 0;
                            } else if (self.data.currentPosition < - thumbnailContainerWidth) {
                                self.data.currentPosition = - thumbnailContainerWidth;
                            }
                            if (horizontalScroll) {
                                thumbnailsContainer.css('left', self.data.currentPosition);
                            } else {
                                thumbnailsContainer.css('top', self.data.currentPosition);
                            }
                        } else {
                            clearInterval(self.thumbnailScrollIntervalRef);
                            self.thumbnailScrollIntervalRef = false;
                        }
                    }, 16);
                }
            },

            stopThumbnailScroll: function (evt) {
                if (this.thumbnailScrollIntervalRef) {
                    clearInterval(this.thumbnailScrollIntervalRef);
                    this.thumbnailScrollIntervalRef = false;
                }
            }
        });


        // Edit JS
        /**
         * Initialize edit mode
         */
        module.LANDSCAPE_OFFSET_PCT = 0.5;
        module.PORTRAIT_OFFSET_PCT = 0.3;
        module.MIN_IMAGE_SIZE_FOR_PAN_EXPLANATION = 110;

        module.safeOneLoaded = function () {
            this.bind("rendered", this._overlayEllipsis);
            if (!webs.photoGalleryLoaded) {
                try {
                    // append style to iframe, not to the parent
                    this.el.closest('html').find('head').first().append('<link rel="stylesheet" type="text/css" href="' + webs.props.staticServer + '/active-static/target/site/modules/photo_gallery/PhotoGalleryLoaderSpinner.css"/>');
                    webs.photoGalleryLoaded = true;
                } catch (ex) {
                    webs.log.error("Unable to setup Photo Gallery's active-static files", this.data, ex);
                }
            }

            var self = this;
            if (this.data.viewType == self.VIEW_TYPE.GRID) {
                this.el.bind("dblclick", function () {
                    self.showSettingsDialog();
                });
            } else if (this.data.viewType == this.VIEW_TYPE.SLIDESHOW) {
                this._loadSlideshow();
            } else {
                throw "Undefined photo gallery view type: " + self.data.viewType;
            }
        };

        module.showSettingsDialog = function () {
            this.toolbar.click("images");
        };

        module.edit = function () {
            this.configureImageControls();
        };

        module.onActivate = function () {
            this.activated = true;
            if (this.data.viewType == this.VIEW_TYPE.GRID) {
                this.setupSortable();
            }
        };

        module.setupSortable = function () {
            if (this.activated && this.data.images.length > 1) {
                var self = this;
                var $placeholder = $(this.el.find('.placeholder-template').html());
                var $moveableContainer = this.el.find('.moveable');

                this.sortableView = new SwappableSortableView({
                    el: $(this.el),
                    $placeholder: $placeholder,
                    $moveableContainer: $moveableContainer,
                    $scrollDiv: self.getScrollableElement(),
                    nonDragSelector: '.w-photo_gallery .photo-settings, ' +
                    '.w-photo_gallery .photo-settings *, ' +
                    '.w-photo_gallery .arrow, ' +
                    '.w-photo_gallery .arrow *, ' +
                    '.w-photo_gallery .container, ' +
                    '.w-photo_gallery .container *, ' +
                    '.w-photo_gallery .webs-gallery-grid-title-caption-social-box, ' +
                    '.w-photo_gallery .webs-gallery-grid-title-caption-social-box *, ' +
                    '.w-photo_gallery .bldr-image-zoomer, ' +
                    '.w-photo_gallery .bldr-image-zoomer *, ' +
                    '.w-photo_gallery .titlecaption-container, ' +
                    '.w-photo_gallery .titlecaption-container *',
                    elementSelector: '.w-photo_gallery .cell',
                    container: this.el.find('.w-photo_gallery'),
                    scrollTarget: 100,
                    scrollSpacingTop: 50,
                    scrollSpacingBottom: 200
                });

                this.sortableView.off('elementReordered');
                this.sortableView.on('elementReordered', function () {
                    var newImagesArray = [];
                    self.getGridElements().each(function (index) {
                        newImagesArray.push(self.data.images[$(this).data('_i_')]);
                        newImagesArray[newImagesArray.length - 1]._i_ = newImagesArray.length - 1;
                        $(this).data('_i_', index);
                    });
                    self.data.images = newImagesArray;
                });

                this.sortableView.start();

                this.sortableView.off('dragStarted');
                this.sortableView.on('dragStarted', function () {
                    self.el.find('.moveable').css(self.determineCellSize());
                    self.deactivateCells(null);
                });
            }
        };

        module.getScrollableElement = function () {
            if (!this.$scrollDiv) {
                var self = this;
                this.$scrollDiv = this.el.parents('body');
                $.each([this.el.parents('body'), this.el.parents('html')], function (index, value) {
                    value.scrollTop(value.scrollTop() + 1);
                    if (value.scrollTop() > 0) {
                        value.scrollTop(value.scrollTop() - 1);
                        self.$scrollDiv = value;
                        return false;
                    }
                });
            }
            return this.$scrollDiv;
        };

        module.configureImageControls = function () {
            this.activePhotoPosition = null;
            for (var i = 0; i < this.data.images.length; i++) {
                this.configurePhotoEditOptions(i);
            }
        };

        module.enableImageControl = function (activePhotoPosition) {
            // Photo that we're currently cropping/zooming
            this.activePhotoPosition = activePhotoPosition;

            // Configure image control specific to our image
            this.imageEditControl = bldr.controls.create('image', {
                container: this.getGridElements(activePhotoPosition).find('.image-edit-wrapper'),
                el: this.getGridElements(activePhotoPosition).find('.webs-image-wrapper-1'),
                data: this.data.images[activePhotoPosition],
                frameWidth: function () {
                    return 0;
                },
                disableImageToolbar: true
            });
            if (this.data.viewType == this.VIEW_TYPE.SLIDESHOW && this.data.settings.slideshow.thumbnailPosition === "thumbnail-bottom") {
                this.el.find(".w-slideshow-thumbnail-area").hide(400);
            }
            this.imageEditControl.show();
        };

        module.getGridElements = function (index) {
            var $cells;
            if (this.data.viewType == this.VIEW_TYPE.GRID) {
                $cells = this.el.find('.w-photo_gallery .cell');
            } else if (this.data.viewType == this.VIEW_TYPE.SLIDESHOW) {
                $cells = this.el.find('.w-photo_gallery .slide');
            } else {
                throw "Undefined photo gallery view type: " + self.data.viewType;
            }

            if (!index && index != 0) {
                return $cells;
            } else {
                return $cells.eq(index);
            }
        };

        module.deactivateCells = function (evt) {
            var clickedPhotoPosition = - 1;
            var self = this;

            if (evt != null) {
                var clickedLink = $(evt.currentTarget);
                if (self.data.viewType == self.VIEW_TYPE.GRID) {
                    var cell = clickedLink.closest('.cell');
                } else if (self.data.viewType == self.VIEW_TYPE.SLIDESHOW) {
                    var cell = clickedLink.closest('.slide');
                } else {
                    throw "Undefined photo gallery view type: " + self.data.viewType;
                }

                clickedPhotoPosition = cell.data('_i_');
            }


            this.getGridElements().each(function (index) {
                if (index != clickedPhotoPosition) {
                    self.deactivateCell(index);
                }
            });
        };

        module.deactivateCell = function (position) {
            if (this.activePhotoPosition == position && this.imageEditControl) {
                if (!this.data && this.data.settings.slideshow.thumbnailPosition === "thumbnail-bottom") {
                    this.el.find(".w-slideshow-thumbnail-area").show(400);
                }
                this.saveFromImageCropper();
                this.imageEditControl.hide();
                this.imageEditControl = null;
            }

            var $cell = this.getGridElements(position);

            if ($cell.hasClass('editing') && this.sortableView) {
                this.sortableView.start();
            }

            $cell.removeClass('editing');
            $cell.find('.image-crop-explanation').hide();
            $cell.find('.titlecaption-container, .arrow').hide();
        };

        module.hideCropOptions = function () {
            for (var i = 0; i < this.data.images.length; i++) {
                var $slide = this.getGridElements(i);
                var $img = $slide.find('.image-container img');
                var clickedPhotoPosition = $slide.data('_i_');
                var $settingsContainer = $slide.find('.photo-settings .crop_photo');
                if ($img.height() <= $slide.height() && $img.width() <= $slide.width()) {
                    $settingsContainer.hide();
                    $slide.find('.photo-settings').addClass("hideCrop");
                } else {
                    $settingsContainer.show();
                    $slide.find('.photo-settings').removeClass("hideCrop");
                }
            }
        };

        module.configurePhotoEditOptions = function (index) {
            var self = this;

            if (self.data.viewType == self.VIEW_TYPE.SLIDESHOW) {
                self.hideCropOptions();
            }

            var $cell = this.getGridElements(index);
            // Allow our title and caption fields to be edited without firing the click event above
            $cell.find('.titlecaption-container').on('dblclick', function (event) {
                event.stopPropagation();
            });

            // Disable all edit menus on cell click
            $cell.on('click', function (evt) {
                evt.stopPropagation();
                self.deactivateCells(evt);
            });

            $cell.find('.w-photo_gallery').on('click', function (evt) {
                self.deactivateCells(null);
            });

            $cell.find('.photo-settings .remove_photo').on('click', function (evt) {
                var $this = $(this);
                _.defer(function () {
                    var $cell = null;
                    if (self.data.viewType == self.VIEW_TYPE.SLIDESHOW) {
                        $cell = $this.closest('.slide');
                    } else {
                        // grid
                        $cell = $this.closest('.cell');
                    }

                    var clickedPhotoPosition = $cell.data('_i_');

                    self.data.images.splice(clickedPhotoPosition, 1);
                    if (self.data.viewType == self.VIEW_TYPE.SLIDESHOW) {
                        self.selectSlide(0, 0);
                    }
                    self.correctThumbnails();
                });
            });

            $cell.find('.photo-settings .crop_photo').on('click', function (evt) {
                var clickedPhotoPosition = null;
                if (self.data.viewType == self.VIEW_TYPE.GRID) {
                    clickedPhotoPosition = $(this).closest('.cell').data('_i_');
                } else if (self.data.viewType == self.VIEW_TYPE.SLIDESHOW) {
                    clickedPhotoPosition = $(this).closest('.slide').data('_i_');
                } else {
                    throw "Undefined photo gallery view type: " + self.data.viewType;
                }

                setTimeout(function () {
                    self.deactivateCell(clickedPhotoPosition);
                    $cell.addClass('editing');
                    self.enableImageControl(clickedPhotoPosition);
                    if (self.sortableView) {
                        self.sortableView.detach();
                    }

                    if (self.data.showCropExplanation && self.data.images && self.data.images[0] && self.data.images[0].width > self.MIN_IMAGE_SIZE_FOR_PAN_EXPLANATION) {
                        self.data.showCropExplanation = false;
                        $cell.find('.image-crop-explanation').show();
                    }
                }, 0);
            });

            $cell.find('.image-crop-explanation').on('mousedown', function (evt) {
                $cell.find('.image-crop-explanation').hide();
                $cell.find('.pannable').trigger(evt);
            });

            $cell.find('.photo-settings .edit_caption').on('click', function (evt) {
                if (self.data.viewType == self.VIEW_TYPE.GRID) {
                    var $cell = $(this).closest('.cell');
                } else if (self.data.viewType == self.VIEW_TYPE.SLIDESHOW) {
                    var $cell = $(this).closest('.slide');
                } else {
                    throw "Undefined photo gallery view type: " + self.data.viewType;
                }

                var clickedPhotoPosition = $cell.data('_i_');

                _.defer(function () {
                    self.deactivateCell(clickedPhotoPosition);
                    $cell.addClass('editing');
                    $cell.find('.titlecaption-container, .arrow').show();

                    $cell.find('.title').on('input keyup change', function () {
                        var title = $.trim($(this).val());
                        self.data.images[clickedPhotoPosition].title = title;

                        if (title === "") {
                            $cell.find('.webs-gallery-grid-title .ellip-content').html('&nbsp;');
                        } else {
                            $cell.find('.webs-gallery-grid-title .ellip-content').text(title);
                        }
                    });

                    $cell.find('.caption').on('input keyup change', function () {
                        var caption = $.trim($(this).val());
                        self.data.images[clickedPhotoPosition].caption = caption;

                        if (caption === "") {
                            $cell.find('.webs-gallery-grid-caption .ellip-content').html('&nbsp;');
                        } else {
                            $cell.find('.webs-gallery-grid-caption .ellip-content').text(caption);
                        }
                    });
                });
            });
        };

        module.onDeactivate = function () {
            this.deactivateCells(null);

            this.toolbar.reset();
            this.activated = false;
            if (this.sortableView) {
                this.sortableView.detach();
                this.sortableView = null;
            }
        };

        module.saveFromImageCropper = function () {
            // Only save if our image cropper is active
            if (this.activePhotoPosition != null && this.imageEditControl) {
                _.extend(this.data.images[this.activePhotoPosition], this.imageEditControl.getData());
            }
        };

        // Get's called every few seconds to save changes to crop/zoom as we are making them
        module.getData = function () {
            this.saveFromImageCropper();
            return this.data;
        };

        // We need our image dimensions, but CSS works them out
        module.determineCellSize = function () {
            var cell = this.el.find('.webs-image-wrapper-1:first')[0];
            if (cell) {
                return {
                    width: cell.offsetWidth,
                    height: cell.offsetHeight
                };
            }
            return;
        };

        module.correctBackgroundColor = function () {
            var slides = this.getGridElements();
            var wrap = $(slides).find(".slide-wrapper");
            wrap.css("background-color", this.data.settings.slideshow.backgroundColor);
        };

        module.configureNewImage = function (imagePosition, cellDimensions, callback) {
            var self = this;
            if (self.data.viewType == self.VIEW_TYPE.GRID) {
                self._configureNewImageGrid(imagePosition, cellDimensions, callback);
            } else if (self.data.viewType == self.VIEW_TYPE.SLIDESHOW) {
                self._configureNewImageSlideshow(imagePosition, cellDimensions, callback);
            } else {
                throw "Undefined photo gallery view type: " + self.data.viewType;
            }
        };

        module.correctThumbnails = function () {
            var self = this;
            if (self.data.viewType == self.VIEW_TYPE.GRID) {
                self._correctThumbnailsGrid();
            } else if (self.data.viewType == self.VIEW_TYPE.SLIDESHOW) {
                self._correctThumbnailsSlideshow();
            } else {
                throw "Undefined photo gallery view type: " + self.data.viewType;
            }
        };

        module.fit = function (width, persist) {
            this.correctThumbnails();
        };

        module.modeChange = function () {
            // deactivate the current module
            this.onDeactivate();

            var cells = this.getGridElements();

            // what?
            for (var i = 0; i < this.data.images.length; i++) {
                var cell = $(cells[i]);
                var cellWrapper = cell.find('.cell-wrapper');
                var rowCellWrappers = cellWrapper.closest('.row').find('.cell-wrapper');
                var rowCellWrapperHeights = _.map(rowCellWrappers, function(cell) {
                    return $(cell).height();
                });
                var captionTitleHeight = cell.find('.webs-gallery-grid-title-caption-social-box').outerHeight(true) || 0;
                var imageDimensions = {
                    width: cellWrapper.width(),
                    height: _.max(rowCellWrapperHeights) - captionTitleHeight
                };
                this.configureNewImage(i, imageDimensions, function () {});
            }

            // simulate the model being loaded and then being clicked on
            this.correctThumbnails();
            this.safeOneLoaded();
            this.onActivate();
            this.attachControls();
        };

        module.attachControls = function () {
            var viewType = this.data.viewType;

            var rampableSlideshowAvailable = webs && webs.site && webs.site.photoGallerySlideshowEnabled;

            if (rampableSlideshowAvailable) {
                if (viewType == this.VIEW_TYPE.GRID) {
                    this.toolbar = bldr.toolbar.create('photo_gallery');
                } else {
                    this.toolbar = bldr.toolbar.create('photo_gallery-slideshow');
                }
            } else {
                // fallback to existing toolbar
                this.toolbar = bldr.toolbar.create('photo_gallery-ramp');
            }

            // Configure and show toolbar
            var toolbarModuleValues = _.extend({
                'images': this.data.images,
                'viewType': viewType
            }, this.data.settings[viewType]);

            this.toolbar.show(toolbarModuleValues);

            if (this.data.settings.grid.deckInspectah) {
                this.toolbar.find('.w-dropdown-container.w-tbui-hoverEffectColor').parent('li')
                .toggle(this.data.settings.grid.hoverEffect !== 'hover-effect-none');
            } else {
                this.toolbar.find('.w-dropdown-container.w-tbui-hoverEffect').parent('li').hide();
                this.toolbar.find('.w-dropdown-container.w-tbui-hoverEffectColor').parent('li').hide();
            }

            var self = this;
            this.toolbar.addListener({
                /* Shared across both grid & slideshow */
                'images': function (data) {
                    self.data.images = data;
                    self.correctThumbnails();
                },
                'viewType': function (val) {
                    self.data.viewType = val;
                    self.modeChange();
                },
                /* Options on grid & slideshow, but specific to each */
                'cropStyle': function (val) {
                    self.data.settings[viewType].cropStyle = val;
                    self.correctThumbnails();
                },
                'showTitle': function (val) {
                    self.data.settings[viewType].showTitle = val;
                    self.correctThumbnails();
                },
                'showCaption': function (val) {
                    self.data.settings[viewType].showCaption = val;
                    self.correctThumbnails();
                },
                'deckInspectah': function (val) {
                    self.data.settings[viewType].deckInspectah = val;
                    var hoverEffectLi = self.toolbar.find('.w-dropdown-container.w-tbui-hoverEffect').parent('li');
                    var hoverEffectColorLi = self.toolbar.find('.w-dropdown-container.w-tbui-hoverEffectColor').parent('li');

                    if (val) {
                        hoverEffectColorLi.toggle(self.data.settings.grid.hoverEffect !== 'hover-effect-none');
                        hoverEffectLi.show();
                    } else {
                        hoverEffectLi.hide();
                        hoverEffectColorLi.hide();
                    }
                },
                /* Toolbar options specific to grid */
                'photosPerRow': function (val) {
                    self.data.settings.grid.photosPerRow = parseInt(val);
                    self.correctThumbnails();
                },
                'gridPadding': function (val) {
                    self.data.settings.grid.gridPadding = val;
                    self.correctThumbnails();
                },
                'hoverEffect': function (val) {
                    self.toolbar.find('.w-dropdown-container.w-tbui-hoverEffectColor').parent('li').toggle(val !== 'hover-effect-none');
                    self.data.settings.grid.hoverEffect = val;
                },
                'hoverEffectColor': function (val) {
                    self.data.settings.grid.hoverEffectColor = val;
                },
                /* Toolbar options specific to slideshow */
                'duration': function (val) {
                    self.data.settings.slideshow.duration = val;
                    self.correctThumbnails();
                },
                'buttonStyle': function (val) {
                    self.data.settings.slideshow.buttonStyle = val;
                    self.correctThumbnails();
                },
                'thumbnailPosition': function (val) {
                    self.data.settings.slideshow.thumbnailPosition = val;
                    self.correctThumbnails();
                },
                'transition': function(val) {
                    self.data.settings.slideshow.transition = val;
                    self.correctThumbnails();
                },
                'backgroundColor': function (val) {
                    self.data.settings.slideshow.backgroundColor = val;
                    self.correctBackgroundColor();
                }
            });
        };

        module._reloadHtml = function () {
            if (this.data.images.length > 0) {
                _.each(this.data.images, function (image) {
                    image.selected = false;
                });
                if ((this.data.currentSlide >= this.data.images.length) || (!this.data.images[this.data.currentSlide])) {
                    this.data.currentSlide = 0;
                }
                this.data.images[this.data.currentSlide].selected = true;
            }

            this.el.html(this.html());
            this.imageViews = [];
            for (var i = 0; i < this.data.images.length; i++) {
                this.imageViews[i] = new PhotoGalleryImageView({
                    model: new PhotoGalleryPhoto(this.data.images[i]),
                    imageReady: false
                });
                this.imageViews[i].render();
                this.getGridElements(i).find('.image-container').append(this.imageViews[i].el);
            }

            if (this.data.viewType == this.VIEW_TYPE.GRID) {
                this.setupSortable();
            }
        };

        /* internal grid-specific code */
        // Make sure a new image is properly scaled to our cell size
        module._configureNewImageGrid = function (imagePosition, cellDimensions, callback) {
            if (this.activated) {
                this.showHelptip(translate('webs.module.photo_gallery.helptip'), {
                    index: 2
                });
            }

            var self = this;
            var image = new Image();

            // set these defaults before the image loads, to prevent broken links in published mode
            this.data.images[imagePosition].module = 'photo_gallery';
            _.extend(self.data.images[imagePosition], cellDimensions);
            _.extend(self.data.images[imagePosition], {
                top: 0,
                left: 0
            });

            image.onload = function () {
                // Landscape (scale vertical to 100% of cell)
                var zoomFactor = cellDimensions.width / image.width;
                if (cellDimensions.height / image.height > zoomFactor) {
                    zoomFactor = cellDimensions.height / image.height;
                }
                var imageWidthPixels = image.width * zoomFactor;

                if (image.width > image.height) {
                    self.data.images[imagePosition].left = Math.floor((cellDimensions.width - imageWidthPixels) * self.LANDSCAPE_OFFSET_PCT);
                } else {
                    var imageHeightPixels = image.height * zoomFactor;
                    self.data.images[imagePosition].top = Math.floor((cellDimensions.height - imageHeightPixels) * self.PORTRAIT_OFFSET_PCT);
                }

                self.data.images[imagePosition].imageWidth = (imageWidthPixels / cellDimensions.width) * 100;

                self.imageViews[imagePosition].changeModel(new PhotoGalleryPhoto(self.data.images[imagePosition]), true);

                self.configurePhotoEditOptions(imagePosition);

                if (callback) 
                    callback();
            };

            image.src = this.data.images[imagePosition].url;
        };

        // Correct panning on thumbnails, used when the size of them changes
        module._correctThumbnailsGrid = function () {

            var self = this;

            // Disable automatic save from our image cropper
            this.activePhotoPosition = null;

            // Render our loading state
            self._reloadHtml();

            var cells = this.getGridElements();

            // Check to make sure all images have been corrected and then render
            var renderView = function () {
                var overlayEllipsisFunc = _.bind(self._overlayEllipsis, self);
                overlayEllipsisFunc();
                self.$('.ellip-anchor').on('click', function (evt) {
                    evt.preventDefault();
                }); // prevent default click handler

                // remove events from parent from prior rendering, to prevent memory leaks (I hope)
                self.el.closest('.webs-bin').off('.photo_gallery');

                // add an event handler to the parent webs-bin for hover, as this changes the bg-color in edit mode
                self.el.closest('.webs-bin').on('mouseenter.photo_gallery', overlayEllipsisFunc);
                self.el.closest('.webs-bin').on('mouseleave.photo_gallery', overlayEllipsisFunc);
            };

            // Adjust position of thumbs since the size changed
            for (var i = 0; i < this.data.images.length; i++) {

                var cell = $(cells[i]);
                var cellWrapper = cell.find('.cell-wrapper');

                var rowCellWrappers = cellWrapper.closest('.row').find('.cell-wrapper');
                var rowCellWrapperHeights = _.map(rowCellWrappers, function(cell) {
                    return $(cell).height();
                });
                var captionTitleHeight = cell.find('.webs-gallery-grid-title-caption-social-box').outerHeight(true) || 0;

                var imageDimensions = {
                    width: cellWrapper.width(),
                    height: _.max(rowCellWrapperHeights) - captionTitleHeight
                };
                var thumbnailResizeFactor = imageDimensions.width / this.data.images[i].width;

                // Correct existing images
                if (this.data.images[i].width) {

                    // Correct position since we changed size
                    this.data.images[i].top = this.data.images[i].top * thumbnailResizeFactor;
                    this.data.images[i].left = this.data.images[i].left * thumbnailResizeFactor;

                    var oldCellDimensions = {
                        width: this.data.images[i].width,
                        height: this.data.images[i].height
                    };
                    this.data.images[i] = _.extend(this.data.images[i], imageDimensions);

                    // Fetch our image dimensions
                    var image = new Image();
                    image.onload = (function (loop) {
                        return function () {

                            var oldImageWidthPixels = Math.round(oldCellDimensions.width * self.data.images[loop].imageWidth / 100);
                            var oldImageHeightPixels = Math.round(oldImageWidthPixels * (this.height / this.width));
                            var missingVerticalSpace = (imageDimensions.height - (oldImageHeightPixels + self.data.images[loop].top));

                            // Check to see if we have enough image to change to square crop
                            if (thumbnailResizeFactor == 1 && imageDimensions.height != oldCellDimensions.height && missingVerticalSpace > 0) {

                                // Detect if we can just shift our image down to get what we need
                                if ((self.data.images[loop].top * - 1) > missingVerticalSpace) {
                                    self.data.images[loop].top += missingVerticalSpace;
                                } else {

                                    // Zoom to get the extra image we need
                                    var newImageWidthPixels = (oldImageWidthPixels * imageDimensions.height) / oldImageHeightPixels;
                                    var zoom = newImageWidthPixels / imageDimensions.width;

                                    self.data.images[loop].top = 0;
                                    self.data.images[loop].imageWidth = (zoom * 100) + 1;
                                }

                            }

                            //remove the loading cover
                            self.imageViews[loop].changeModel(new PhotoGalleryPhoto(self.data.images[loop]), true);
                            self.configurePhotoEditOptions(loop);

                            renderView();
                        }
                    }(i));

                    image.src = this.data.images[i].url;

                } else {
                    this.configureNewImage(i, imageDimensions, function () {
                        renderView();
                    });
                }
            }
        };

        /* Internal slideshow-specific code */
        module._loadSlideshow = function() {
            var self = this;
            _.defer(function() {
                self.correctBackgroundColor();
                self.setupSlideshow();
            });
            this.data.currentPosition = 0;
            this.el.on('click', '.thumbnail', _.bind(self.thumbnailClick, self));
            this.el.on('mouseenter', '.w-slideshow-thumbnail-button', _.bind(self.thumbnailScroll, self));
            this.el.on('mouseleave', '.w-slideshow-thumbnail-button', _.bind(self.stopThumbnailScroll, self));

            //set the selected slide to the first one
            for (var i = 0; i < self.data.images.length; i++) {
                this.el.find(".webs-gallery-grid-title-caption-social-box[data-_i_=" + i + "]").removeClass("slide-selected");
                this.el.find(".slide-wrapper[data-_i_=" + i + "]").removeClass("slide-selected");
                this.el.find(".thumbnail[data-_i_=" + i + "]").removeClass("slide-selected");
                this.data.images[i].selected = false;
            }

            this.el.find(".webs-gallery-grid-title-caption-social-box[data-_i_=0]").addClass("slide-selected");
            this.el.find(".slide-wrapper[data-_i_=0]").addClass("slide-selected");
            this.el.find(".thumbnail[data-_i_=0]").addClass("slide-selected");
            this.data.images[0].selected = true;

            /* Handle double clicks in a safe manner*/
            var mouseInButton = false;
            this.el.on('mouseenter', '.button', function (evt) {
                mouseInButton = true;
            });
            this.el.on('mouseleave', '.button', function (evt) {
                mouseInButton = false;
            });

            this.el.on("dblclick", '.slide, .bldr-placeholder-photo-gallery', function () {
                if (!mouseInButton) {
                    self.showSettingsDialog();
                }
            });
        };

        module._configureNewImageSlideshow = function (imagePosition, slideDimensions, callback) {
            if (this.activated) {
                this.showHelptip(translate('webs.module.photo_gallery.helptip'), {
                    index: 2
                });
            }

            var self = this;
            var image = new Image();

            // set these defaults before the image loads, to prevent broken links in published mode
            this.data.images[imagePosition].module = 'photo_gallery';
            _.extend(self.data.images[imagePosition], slideDimensions);
            _.extend(self.data.images[imagePosition], {
                top: 0,
                left: 0
            });

            image.onload = function () {

                var zoomFactor = slideDimensions.width / image.width;
                var imageHeightPixels = image.height * zoomFactor;
                var imageWidthPixels = image.width * zoomFactor;

                //image fits perfectly: center + bg color
                if (image.width <= slideDimensions.width && image.height <= slideDimensions.height) {
                    self.data.images[imagePosition].left = Math.floor((slideDimensions.width - image.width) * 0.5);
                    self.data.images[imagePosition].top = Math.floor((slideDimensions.height - image.height) * 0.5);
                    self.data.images[imagePosition].imageWidth = (image.width / slideDimensions.width) * 100;
                }
                //shrink image width to be = slide width
                else if (image.width > slideDimensions.width && image.height > slideDimensions.height && image.width <= image.height) {
                    self.data.images[imagePosition].imageWidth = 100;
                    self.data.images[imagePosition].left = 0;
                }
                //shrink image height to be = slide height
                else if (image.width > slideDimensions.width && image.height > slideDimensions.height && image.width > image.height) {
                    var aspectRatio = image.width / image.height;
                    self.data.images[imagePosition].top = 0;
                    self.data.images[imagePosition].imageWidth = ((slideDimensions.height * aspectRatio) / slideDimensions.width) * 100;
                }
                //shrink image width to be = slide width. center image vertically
                else if (image.width > slideDimensions.width && image.height <= slideDimensions.height) {
                    self.data.images[imagePosition].imageWidth = 100;
                    self.data.images[imagePosition].top = Math.floor((slideDimensions.height - imageHeightPixels) * 0.5);
                    self.data.images[imagePosition].left = 0;
                }
                //shrink image width such that image height = slide height
                else if (image.height > slideDimensions.height && image.width <= slideDimensions.width) {
                    self.data.images[imagePosition].top = 0;
                    var aspectRatio = image.width / image.height;
                    self.data.images[imagePosition].imageWidth = ((slideDimensions.height * aspectRatio) / slideDimensions.width) * 100;
                    self.data.images[imagePosition].left = Math.floor(((100 - self.data.images[imagePosition].imageWidth) / 200) * slideDimensions.width);

                }

                self.imageViews[imagePosition].changeModel(new PhotoGalleryPhoto(self.data.images[imagePosition]), true);

                self.configurePhotoEditOptions(imagePosition);

                if (callback) 
                    callback();
            };

            image.src = this.data.images[imagePosition].url;
        };

        module._correctThumbnailsSlideshow = function () {
            var self = this;
            // Disable automatic save from our image cropper
            this.activePhotoPosition = null;

            // Set our heights to false so we render the table naturally
            var oldHeights = [];
            for (var i = 0; i < this.data.images.length; i++) {
                oldHeights.push(this.data.images[i].height);
                this.data.images[i].height = false;
            }

            // Render so we can get our new thumbnail width
            self._reloadHtml();

            // Set our heights back to normal
            for (var i = 0; i < this.data.images.length; i++) {
                this.data.images[i].height = oldHeights[i];
            }

            var slides = this.getGridElements();

            // Check to make sure all images have been corrected and then render
            var renderView = function () {
                var overlayEllipsisFunc = _.bind(self._overlayEllipsis, self);
                overlayEllipsisFunc();
                self.$('.ellip-anchor').on('click', function (evt) {
                    evt.preventDefault();
                }); // prevent default click handler

                // remove events from parent from prior rendering, to prevent memory leaks (I hope)
                self.el.closest('.webs-bin').off('.photo_gallery');

                // add an event handler to the parent webs-bin for hover, as this changes the bg-color in edit mode
                self.el.closest('.webs-bin').on('mouseenter.photo_gallery', overlayEllipsisFunc);
                self.el.closest('.webs-bin').on('mouseleave.photo_gallery', overlayEllipsisFunc);
            };


            var slideWrapper = $(slides[0]).find(".slide-wrapper");
            for (var i = 0; i < self.data.images.length; i++) {
                var wrap = $(slides[i]).find(".slide-wrapper");
                if (wrap.hasClass("slide-selected")) {
                    slideWrapper = wrap;
                    break;
                }
            }

            // Adjust position of thumbs since the size changed
            for (var i = 0; i < this.data.images.length; i++) {
                var slide = $(slides[i]);
                var slideDimensions = {};
                var captionTitleHeight = slide.find('.webs-gallery-grid-title-caption-social-box').outerHeight(true) || 0;

                slideDimensions.width = slideWrapper.width();
                slideDimensions.height = slideWrapper.height() - captionTitleHeight;

                var thumbnailResizeFactor = 1;

                // Correct existing images
                if (this.data.images[i].width) {

                    var oldSlideDimensions = {
                        width: this.data.images[i].width,
                        height: this.data.images[i].height
                    };
                    this.data.images[i] = _.extend(this.data.images[i], slideDimensions);

                    // Fetch our image dimensions
                    var image = new Image();
                    image.onload = (function (loop) {
                        return function () {
                            var zoomFactor = slideDimensions.width / image.width;
                            var imageHeightPixels = this.height * zoomFactor;
                            var imageWidthPixels = this.width * zoomFactor;

                            //image fits perfectly: center + bg color
                            if (this.width <= slideDimensions.width && this.height <= slideDimensions.height) {
                                self.data.images[loop].left = Math.floor((slideDimensions.width - this.width) * 0.5);
                                self.data.images[loop].top = Math.floor((slideDimensions.height - this.height) * 0.5);
                                self.data.images[loop].imageWidth = (this.width / slideDimensions.width) * 100;
                            }
                            //shrink image width to be = slide width
                            else if (this.width > slideDimensions.width && this.height > slideDimensions.height && this.width <= this.height) {
                                self.data.images[loop].imageWidth = 100;
                                self.data.images[loop].left = 0;
                            }
                            //shrink this height to be = slide height
                            else if (this.width > slideDimensions.width && this.height > slideDimensions.height && this.width > this.height) {
                                var aspectRatio = this.width / this.height;
                                self.data.images[loop].top = 0;
                                self.data.images[loop].imageWidth = ((slideDimensions.height * aspectRatio) / slideDimensions.width) * 100;
                            }
                            //shrink image width to be = slide width. center image vertically
                            else if (this.width > slideDimensions.width && this.height <= slideDimensions.height) {
                                self.data.images[loop].imageWidth = 100;
                                self.data.images[loop].top = Math.floor((slideDimensions.height - imageHeightPixels) * 0.5);
                                self.data.images[loop].left = 0;
                            }
                            //shrink image width such that image height = slide height
                            else if (this.height > slideDimensions.height && this.width <= slideDimensions.width) {
                                self.data.images[loop].top = 0;
                                var aspectRatio = image.width / image.height;
                                self.data.images[loop].imageWidth = ((slideDimensions.height * aspectRatio) / slideDimensions.width) * 100;
                                self.data.images[loop].left = Math.floor(((100 - self.data.images[loop].imageWidth) / 200) * slideDimensions.width);
                            }

                            self.imageViews[loop].changeModel(new PhotoGalleryPhoto(self.data.images[loop]), true);

                            //remove the loading cover
                            self.imageViews[loop].removeLoadingCover();
                            self.configurePhotoEditOptions(loop);

                            renderView();
                        }
                    }(i));

                    image.src = this.data.images[i].url;

                } else {
                    this.configureNewImage(i, slideDimensions, function () {
                        renderView();
                    });
                }
            }

            var thumbnailArea = self.el.find(".w-slideshow-thumbnail-area");
            var thumbnails = self.el.find(".w-slideshow-thumbnails-ul");
            self.data.displayThumbnailButton = (thumbnails.width() > thumbnailArea.width()) || (thumbnails.height() > thumbnailArea.height());

            self.data.displaySlideButton = self.data.images.length > 1;

            // Force edit to update
            self._reloadHtml();
        };

    })(mod, ext);

    return ModuleClassLoader.register('photo_gallery', mod, ext);
});




// photo_gallery_slideshow Module (custom)

define('module/photo_gallery_slideshow', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.photo_gallery_slideshow', 'underscore', 'backbone', 'site/deckInspectah/DeckInspectahView', 'site/modules/photo_gallery/PhotoGalleryDeckInspectahImageView', 'site/modules/photo_gallery/PhotoGalleryDeckInspectahSidebarView', 'site/modules/photo_gallery/PhotoGalleryPhoto', 'site/modules/photo_gallery/SwappableSortableView', 'site/modules/photo_gallery/PhotoGalleryImageView'],
function($, ModuleClassLoader, translate, _, Backbone, DeckInspectahView, PhotoGalleryDeckInspectahImageView, PhotoGalleryDeckInspectahSidebarView, PhotoGalleryPhoto, SwappableSortableView, PhotoGalleryImageView) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="moveable"/>\
        <div class="w-photo_gallery {{settings.slideshow.cropStyle}}" style="height: {{containerHeight}};">\
        {{#if images.length}}\
        <div class="button left-button {{settings.slideshow.buttonStyle}}"></div>\
        <div class="button right-button {{settings.slideshow.buttonStyle}}"></div>\
        {{/if}}\
        {{#each images}}\
        <div class="slide{{#if selected}} selected{{/if}} {{../settings.slideshow.transition}}" data-_i_="{{_i_}}">\
        <div class="slide-wrapper">\
        <div class="photo-settings">\
        <a class="crop_photo" title="{{#l}}webs.module.photo_gallery.cropPhoto{{/l}}">\
        <img src="http://static.websimages.com/static/projects/finch/images/photo_gallery-crop.png">\
        </a>\
        <a class="edit_caption" title="{{#l}}webs.module.photo_gallery.dialog.editCaption{{/l}}">\
        <img src="http://static.websimages.com/static/projects/finch/images/photo_gallery-comment.png">\
        </a>\
        <a class="remove_photo last" title="{{#l}}webs.module.photo_gallery.dialog.removePhoto{{/l}}">\
        <img src="http://static.websimages.com/static/projects/finch/images/photo_gallery-delete.png">\
        </a>\
        </div>\
        \
        <div class="arrow">\
        </div>\
        <div class="titlecaption-container">\
        <div class="inner-titlecaption-container">\
        <input class="title" type="text"\
        placeholder="{{#l}}webs.module.photo_gallery.dialog.setting.titlePlaceholder{{/l}}"\
        value="{{this.title}}"/>\
        <textarea class="caption"\
        placeholder="{{#l}}webs.module.photo_gallery.dialog.setting.captionPlaceholder{{/l}}">{{this.caption}}</textarea>\
        </div>\
        </div>\
        \
        <div class="image-container"></div>\
        {{#if ../settings.slideshow.showTitle}}\
        <div class="webs-gallery-grid-title-caption-social-box w-font-lato">\
        <div class="webs-gallery-grid-title ellip-container">\
        <div class="ellip-wrap">\
        <p class="ellip-content">{{#if this.title}}{{this.title}}{{else}}\
        &nbsp;{{/if}}</p>\
        </div>\
        <span class="ellipsis">\
        <a href="#" title="{{#l}}webs.module.photo_gallery.showLightbox{{/l}}"\
        class="ellip-anchor">&hellip;</a>\
        </span>\
        </div>\
        <div class="webs-gallery-grid-caption ellip-container">\
        <div class="ellip-wrap">\
        <p class="ellip-content">{{#if this.caption}}{{this.caption}}{{else}}\
        &nbsp;{{/if}}</p>\
        </div>\
        <span class="ellipsis">\
        <a href="#" title="{{#l}}webs.module.photo_gallery.showLightbox{{/l}}"\
        class="ellip-anchor">&hellip;</a>\
        </span>\
        </div>							\
        </div>\
        {{/if}}\
        </div>\
        </div>\
        {{/each}}\
        \
        {{#unless images.length}}\
        <div class="bldr-placeholder-photo-gallery bldr-placeholder-element">\
        <span class="placeholder-text">{{#l}}webs.module.photo_gallery.view.noImage{{/l}}</span>\
        </div>\
        {{/unless}}\
        </div>\
        </div>\
        \
        <div class="placeholder-template" style="display: none">\
        <div class="slide">\
        <div class="slide-wrapper">\
        <div class="image-edit-wrapper">\
        <div class="webs-image-wrapper-1">\
        <div class="webs-image-crop">\
        <div class="placeholderText">\
        {{#l}}webs.module.photo_gallery.placeholder.text{{/l}}\
        </div>\
        </div>\
        </div>\
        </div>\
        </div>\
        </div>\
        </div>\
        ';

        // Module Default Data
        extend.defaultData = {
            "type": "slideshow",
            "showCropExplanation": true,
            "images": [],
            "settings": {
                "grid": {
                    "photosPerRow": 3,
                    "gridPadding": "padding-medium",
                    "cropStyle": "crop-style-square",
                    "deckInspectah": true,
                    "showTitle": true,
                    "showCaption": "never",
                    "hoverEffect": "hover-effect-zoomSign",
                    "hoverEffectColor": "hover-effect-color-black"
                },
                "slideshow": {
                    "duration": 3000,
                    "buttonStyle": 1,
                    "cropStyle": "crop-style-square",
                    "showTitle": true,
                    "deckInspectah": true,
                    "showCaption": "never",
                    "transition": "slide-fade"
                }
            }
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "base": {
                "global": {
                    "css": "view.less"
                },
                "slug": "base"
            }
        };
        if (!extend.styles['base']['global']) {
            extend.styles['base']['global'] = {};
        }
        extend.styles['base']['global']['js'] = null;
        extend.defaultStyle = extend.styles['base'];

        // Register Toolbars
        bldr.toolbar.register('photo_gallery_slideshow', {
            "icon": "photo_gallery",
            "items": [{
                "label": "photo_gallery_slideshow.manageImages",
                "name": "images",
                "dialog": {
                    "url": "/s/modules/photo_gallery_slideshow/v1.23/photo_gallery_slideshow.dialog.html",
                    "heading": "Slideshow Settings",
                    "localizeHeading": "webs.module.photo_gallery_slideshow.dialog.header",
                    "width": 750,
                    "height": 500
                },
                "style": "blue"
            }, {
                "_comment": "Should be refactored to use webs.util.Settings and modify localizeToolbarText() in toolbar.js",
                "type": "menu",
                "label": "photo_gallery_slideshow.settings",
                "menuLabel": "Slideshow Settings",
                "menu": [{
                    "display": "inline",
                    "label": "Autoplay",
                    "type": "dropdown",
                    "name": "duration",
                    "width": 100,
                    "options": [{
                        "label": "Off",
                        "value": 0
                    }, {
                        "label": "1 Second",
                        "value": 1000
                    }, {
                        "label": "2 Seconds",
                        "value": 2000
                    }, {
                        "label": "3 Seconds",
                        "value": 3000
                    }, {
                        "label": "4 Seconds",
                        "value": 4000
                    }, {
                        "label": "5 Seconds",
                        "value": 5000
                    }
                    ]
                }, {
                    "display": "inline",
                    "label": "Transition",
                    "type": "dropdown",
                    "name": "transition",
                    "width": 100,
                    "options": [{
                        "label": "None",
                        "value": 0
                    }, {
                        "label": "Fade",
                        "value": "slide-fade"
                    }
                    ]
                }, {
                    "display": "inline",
                    "label": "photo_gallery.cropStyle",
                    "type": "dropdown",
                    "name": "cropStyle",
                    "width": 100,
                    "options": [{
                        "label": "photo_gallery_slideshow.cropStyleSquare",
                        "value": "crop-style-square"
                    }, {
                        "label": "photo_gallery_slideshow.cropStyleRectangle",
                        "value": "crop-style-rectangle"
                    }, {
                        "label": "photo_gallery_slideshow.cropStyleWidescreen",
                        "value": "crop-style-widescreen"
                    }, {
                        "label": "photo_gallery_slideshow.cropStylePortrait",
                        "value": "crop-style-portrait"
                    }
                    ]
                }, {
                    "display": "inline",
                    "label": "photo_gallery_slideshow.buttonStyle",
                    "type": "dropdown",
                    "name": "buttonstyle",
                    "width": 100,
                    "options": [{
                        "label": "1",
                        "value": 1
                    }
                    ]
                }, {
                    "type": "switch-bare",
                    "label": "photo_gallery.title",
                    "name": "showTitle",
                    "display": "inline"
                }
                ]
            }
            ]
        });

        // View JS
        $.extend(module, {

            /**
            	 * Init function
            	 */
            oneLoaded: function () {
                var self = this;

                if ($(".w-photo_gallery_slideshow-mobile").length) {
                    this.mobileSetup(self);
                }


                // if we haven't already included the deckInspectah css, do so now, and set a global variable so we don't try to include it again
                if (!webs.deckInspectah) {
                    try {
                        $("head").append('<link rel="stylesheet" type="text/css" href="' + webs.props.staticServer + '/active-static/target/site/deckInspectah/deckInspectah.css"/>');
                        webs.deckInspectah = true;
                    } catch (ex) {
                        webs.log.error("Unable to setup Deck Inspectah", this.data, ex);
                    }
                }

                var duration = self.data.settings.slideshow.duration;
                var autoplay = parseInt(duration, 10) != 0;
                if (autoplay && self.data.images.length > 1) {
                    this.autoplay(duration);
                }

                this.el.on('click', '.ellip-anchor, .deck-inspectah-link', function (evt) {
                    evt.preventDefault();

                    var gridSettings = self.data.settings.grid;
                    var isEllipAnchor = $(evt.target).hasClass('ellip-anchor');

                    var closestCell = $(this).closest('.slide');
                    var isMobile = closestCell.hasClass('mobile-image');
                    var image = closestCell.data('_i_');
                    var titleEnabled = (!isMobile);
                    var captionEnabled = (!isMobile);

                    var deckInspectahView = new DeckInspectahView({
                        model: new Backbone.Collection(self.data.images),
                        selectedImage: image,
                        imageViewClass: PhotoGalleryDeckInspectahImageView,
                        sidebarViewClass: PhotoGalleryDeckInspectahSidebarView,
                        titleEnabled: titleEnabled,
                        captionEnabled: captionEnabled
                    });
                    $('body').append(deckInspectahView.el);
                    deckInspectahView.show();
                });

                var overlayEllipsis = _.bind(this._overlayEllipsis, this);
                try {
                    if (top && top.dsnr) {
                        top.require(['internal/sitebuilder/designerChrome/dsnrLoader'], function (dsnrLoader) {
                            dsnrLoader.sitePreview.bind('iframeRestyled', overlayEllipsis)
                        });
                    }
                } catch (e) {
                    // No designer - we can safely ignore
                }

                // always call the deferred, in case we are not on a designer frame
                _.defer(overlayEllipsis);

                this.data.currentSlide = 0;
                this.setupSlideshow(this);
            },

            mobileSetup: function(context) {
                $(".w-photo_gallery_slideshow-mobile").css("height", $(".slide.selected").height());
                context.data.settings.slideshow.duration = 3000;
                $(".progress-timer").addClass("move");
            },


            _overlayEllipsis: function () {
                var self = this;
                // ignore elements that aren't showing title & caption
                if (this.$('.ellip-content')[0]) {
                    var parents = self.$('.ellip-content').eq(0).parents();
                    var bgColor = _.reduce(parents.toArray().reverse(), function (left, el) {
                        var rgba = $(el).css('background-color').match(/[\d\.]+/g);

                        // handle no background - webkit returns 0,0,0,0, gecko returns null
                        if (rgba && rgba.length < 4) {
                            //if there are only 1 (IE) or 3 (webkit/gecko) colors, then it's fully opaque
                            return rgba;
                        } else if (rgba && rgba[3] != 0) {
                            // ignore transparent backgrounds, otherwise do alpha blending
                            return _.map(rgba, function (val, key, rgba) {
                                return key === 3 ? 1 : Math.round(rgba[3] * val + (1 - rgba[3]) * left[key]);
                            });
                        } else {
                            // transparent, return the previous background
                            return left;
                        }
                    }, [255, 255, 255, 0]); // start with a white background

                    bgColor[3] = bgColor[3] || 1;

                    var rgbaStr = 'rgba(' + bgColor.toString() + ')';
                    var rgb = bgColor.slice(0, 3).toString(); // not creating rgbString as we need both rgba(x,x,x,0) and rgb(x,x,x)

                    this.$('.ellipsis').css('background-image',
                    'linear-gradient(to right, rgba(' + rgb + ',0), ' + rgbaStr + ' 50%, ' + rgbaStr + ')');

                    this.$('.ellip-anchor').css('background-color', 'rgb(' + rgb + ')');
                }
            },
            autoplay: function (duration) {
                var self = this;
                var progressbar = $(".progress-timer");
                setInterval(function () {
                    progressbar.toggleClass("move");
                    setTimeout(function() {
                        progressbar.toggleClass("move");
                    }, 1);
                    self.nextSlide();
                }, duration);
            },

            setupSlideshow: function(context) {
                context.data.currentSlide = 0;

                for (var i = 0; i < context.data.images.length; i++) {
                    if (context.data.images[i].selected) {
                        context.data.currentSlide = i;
                    }
                }

                context.el.on('mousedown', '.right-button', function (evt) {
                    context.nextSlide(evt);
                });

                context.el.on('mousedown', '.left-button', function (evt) {
                    context.prevSlide(evt);
                });
            },

            nextSlide: function () {
                if (this.data.images && this.data.images.length > 1) {
                    var prev = this.data.currentSlide;
                    var next = this.data.currentSlide + 1;
                    if (next == this.data.images.length) {
                        next = 0;
                    }
                    this.selectSlide(prev, next);
                }
            },

            prevSlide: function () {
                if (this.data.images && this.data.images.length > 1) {
                    var prev = this.data.currentSlide;
                    var next = this.data.currentSlide - 1;
                    if (next == - 1) {
                        next = this.data.images.length - 1;
                    }
                    this.selectSlide(prev, next);
                }
            },

            selectSlide: function(prev, next) {
                this.data.images[prev].selected = false;
                this.el.find(".slide[data-_i_=" + prev + "]").removeClass("selected");
                this.el.find(".slide[data-_i_=" + next + "]").addClass("selected");
                this.data.currentSlide = next;
                this.data.images[next].selected = true;
            }
        });


        // Edit JS
        /**
         * Initialize edit mode
         */
        module.LANDSCAPE_OFFSET_PCT = 0.5;
        module.PORTRAIT_OFFSET_PCT = 0.3;
        module.MIN_IMAGE_SIZE_FOR_PAN_EXPLANATION = 110;

        module.safeOneLoaded = function () {
            this.bind("rendered", this._overlayEllipsis);
            if (!webs.photoGalleryLoaded) {
                try {
                    // append style to iframe, not to the parent
                    this.el.closest('html').find('head').first().append('<link rel="stylesheet" type="text/css" href="' + webs.props.staticServer + '/active-static/target/site/modules/photo_gallery/PhotoGalleryLoaderSpinner.css"/>');
                    webs.photoGalleryLoaded = true;
                } catch (ex) {
                    webs.log.error("Unable to setup Photo Gallery's active-static files", this.data, ex);
                }
            }

            var self = this;
            setTimeout(function() {
                self.setupSlideshow(self);
            }, 1);
        };

        module.showSettingsDialog = function () {
            this.toolbar.click("images");
        };

        module.edit = function () {

            this.configureImageControls();
        };

        module.onActivate = function () {
            this.activated = true;
            if (this.data.type == "grid") {
                this.setupSortable();
            }
        };

        module.setupSortable = function () {
            if (this.activated && this.data.images.length > 1) {
                var self = this;
                var $placeholder = $(this.el.find('.placeholder-template').html());
                var $moveableContainer = this.el.find('.moveable');

                this.sortableView = new SwappableSortableView({
                    el: $(this.el),
                    $placeholder: $placeholder,
                    $moveableContainer: $moveableContainer,
                    $scrollDiv: self.getScrollableElement(),
                    nonDragSelector: '.w-photo_gallery .photo-settings, ' +
                    '.w-photo_gallery .photo-settings *, ' +
                    '.w-photo_gallery .arrow, ' +
                    '.w-photo_gallery .arrow *, ' +
                    '.w-photo_gallery .container, ' +
                    '.w-photo_gallery .container *, ' +
                    '.w-photo_gallery .webs-gallery-grid-title-caption-social-box, ' +
                    '.w-photo_gallery .webs-gallery-grid-title-caption-social-box *, ' +
                    '.w-photo_gallery .bldr-image-zoomer, ' +
                    '.w-photo_gallery .bldr-image-zoomer *, ' +
                    '.w-photo_gallery .titlecaption-container, ' +
                    '.w-photo_gallery .titlecaption-container *',
                    elementSelector: '.w-photo_gallery .slide',
                    container: this.el.find('.w-photo_gallery'),
                    scrollTarget: 100,
                    scrollSpacingTop: 50,
                    scrollSpacingBottom: 200
                });

                this.sortableView.off('elementReordered');
                this.sortableView.on('elementReordered', function () {
                    var newImagesArray = [];
                    self.getGridElements().each(function (index) {
                        newImagesArray.push(self.data.images[$(this).data('_i_')]);
                        newImagesArray[newImagesArray.length - 1]._i_ = newImagesArray.length - 1;
                        $(this).data('_i_', index);
                    });
                    self.data.images = newImagesArray;
                });

                this.sortableView.start();

                this.sortableView.off('dragStarted');
                this.sortableView.on('dragStarted', function () {
                    self.el.find('.moveable').css(self.determineCellSize());
                    self.deactivateCells(null);
                });
            }
        };

        module.getScrollableElement = function () {
            if (!this.$scrollDiv) {
                var self = this;
                this.$scrollDiv = this.el.parents('body');
                $.each([this.el.parents('body'), this.el.parents('html')], function (index, value) {
                    value.scrollTop(value.scrollTop() + 1);
                    if (value.scrollTop() > 0) {
                        value.scrollTop(value.scrollTop() - 1);
                        self.$scrollDiv = value;
                        return false;
                    }
                });
            }
            return this.$scrollDiv;
        };

        module.configureImageControls = function () {
            this.activePhotoPosition = null;
            for (var i = 0; i < this.data.images.length; i++) {
                this.configurePhotoEditOptions(i);
            }
        };

        module.enableImageControl = function (activePhotoPosition) {
            // Photo that we're currently cropping/zooming
            this.activePhotoPosition = activePhotoPosition;

            // Configure image control specific to our image
            this.imageEditControl = bldr.controls.create('image', {
                container: this.getGridElements(activePhotoPosition).find('.image-edit-wrapper'),
                el: this.getGridElements(activePhotoPosition).find('.webs-image-wrapper-1'),
                data: this.data.images[activePhotoPosition],
                frameWidth: function () {
                    return 0;
                },
                disableImageToolbar: true
            });

            this.imageEditControl.show();
        };

        module.getGridElements = function (index) {
            var $slides = this.el.find('.w-photo_gallery .slide');
            if (!index && index != 0) {
                return $slides;
            } else {
                return $slides.eq(index);
            }
        };

        module.deactivateCells = function (evt) {
            var clickedPhotoPosition = - 1;
            if (evt != null) {
                var clickedLink = $(evt.currentTarget);
                var slide = clickedLink.closest('.slide');
                clickedPhotoPosition = slide.data('_i_');
            }

            var self = this;
            this.getGridElements().each(function (index) {
                if (index != clickedPhotoPosition) {
                    self.deactivateCell(index);
                }
            });
        };

        module.deactivateCell = function (position) {
            if (this.activePhotoPosition == position && this.imageEditControl) {
                this.saveFromImageCropper();
                this.imageEditControl.hide();
                this.imageEditControl = null;
            }

            var $slide = this.getGridElements(position);

            if ($slide.hasClass('editing') && this.sortableView) {
                this.sortableView.start();
            }

            $slide.removeClass('editing');
            $slide.find('.image-crop-explanation').hide();
            $slide.find('.titlecaption-container, .arrow').hide();
        };

        module.configurePhotoEditOptions = function (index) {
            var self = this;

            var $slide = this.getGridElements(index);
            // Allow our title and caption fields to be edited without firing the click event above
            $slide.find('.titlecaption-container').on('dblclick', function (event) {
                event.stopPropagation();
            });

            // Disable all edit menus on slide click
            $slide.on('click', function (evt) {
                evt.stopPropagation();
                self.deactivateCells(evt);
            });

            $slide.find('.w-photo_gallery').on('click', function (evt) {
                self.deactivateCells(null);
            });

            $slide.find('.photo-settings .remove_photo').on('click', function (evt) {
                var $this = $(this);
                _.defer(function () {
                    var slide = $this.closest('.slide');
                    var clickedPhotoPosition = slide.data('_i_');

                    self.data.images.splice(clickedPhotoPosition, 1);
                    self.correctThumbnails();
                });
            });

            $slide.find('.photo-settings .crop_photo').on('click', function (evt) {
                var $slide = $(this).closest('.slide');
                var clickedPhotoPosition = $slide.data('_i_');

                setTimeout(function () {
                    self.deactivateCell(clickedPhotoPosition);
                    $slide.addClass('editing');
                    self.enableImageControl(clickedPhotoPosition);
                    if (self.sortableView) {
                        self.sortableView.detach();
                    }

                    if (self.data.showCropExplanation && self.data.images && self.data.images[0] && self.data.images[0].width > self.MIN_IMAGE_SIZE_FOR_PAN_EXPLANATION) {
                        self.data.showCropExplanation = false;
                        $slide.find('.image-crop-explanation').show();
                    }
                }, 0);
            });

            $slide.find('.image-crop-explanation').on('mousedown', function (evt) {
                $slide.find('.image-crop-explanation').hide();
                $slide.find('.pannable').trigger(evt);
            });

            $slide.find('.photo-settings .edit_caption').on('click', function (evt) {
                var $slide = $(this).closest('.slide');
                var clickedPhotoPosition = $slide.data('_i_');

                _.defer(function () {
                    self.deactivateCell(clickedPhotoPosition);
                    $slide.addClass('editing');
                    $slide.find('.titlecaption-container, .arrow').show();

                    $slide.find('.title').on('input keyup change', function () {
                        var title = $.trim($(this).val());
                        self.data.images[clickedPhotoPosition].title = title;

                        if (title === "") {
                            $slide.find('.webs-gallery-grid-title .ellip-content').html('&nbsp;');
                        } else {
                            $slide.find('.webs-gallery-grid-title .ellip-content').text(title);
                        }
                    });

                    $slide.find('.caption').on('input keyup change', function () {
                        var caption = $.trim($(this).val());
                        self.data.images[clickedPhotoPosition].caption = caption;

                        if (caption === "") {
                            $slide.find('.webs-gallery-grid-caption .ellip-content').html('&nbsp;');
                        } else {
                            $slide.find('.webs-gallery-grid-caption .ellip-content').text(caption);
                        }
                    });
                });
            });
        };

        module.onDeactivate = function () {
            this.deactivateCells(null);

            this.toolbar.reset();
            this.activated = false;
            if (this.sortableView) {
                this.sortableView.detach();
                this.sortableView = null;
            }
        };

        module.saveFromImageCropper = function () {
            // Only save if our image cropper is active
            if (this.activePhotoPosition != null && this.imageEditControl) {
                _.extend(this.data.images[this.activePhotoPosition], this.imageEditControl.getData());
            }
        };

        // Get's called every few seconds to save changes to crop/zoom as we are making them
        module.getData = function () {
            this.saveFromImageCropper();
            return this.data;
        };

        // We need our image dimensions, but CSS works them out
        module.determineCellSize = function () {
            var slide = this.el.find('.webs-image-wrapper-1:first')[0];
            if (slide) {
                return {
                    width: slide.offsetWidth,
                    height: slide.offsetHeight
                };
            }
            return;
        };

        // Make sure a new image is properly scaled to our slide size
        module.configureNewImage = function (imagePosition, slideDimensions, callback) {
            if (this.activated) {
                this.showHelptip(translate('webs.module.photo_gallery.helptip'), {
                    index: 2
                });
            }

            var self = this;
            var image = new Image();

            // set these defaults before the image loads, to prevent broken links in published mode
            this.data.images[imagePosition].module = 'photo_gallery';
            _.extend(self.data.images[imagePosition], slideDimensions);
            _.extend(self.data.images[imagePosition], {
                top: 0,
                left: 0
            });

            image.onload = function () {
                // Landscape (scale vertical to 100% of slide)
                var zoomFactor = slideDimensions.width / image.width;
                if (slideDimensions.height / image.height > zoomFactor) {
                    zoomFactor = slideDimensions.height / image.height;
                }
                var imageWidthPixels = image.width * zoomFactor;

                if (image.width > image.height) {
                    self.data.images[imagePosition].left = Math.floor((slideDimensions.width - imageWidthPixels) * self.LANDSCAPE_OFFSET_PCT);
                } else {
                    var imageHeightPixels = image.height * zoomFactor;
                    self.data.images[imagePosition].top = Math.floor((slideDimensions.height - imageHeightPixels) * self.PORTRAIT_OFFSET_PCT);
                }

                self.data.images[imagePosition].imageWidth = (imageWidthPixels / slideDimensions.width) * 100;

                self.imageViews[imagePosition].changeModel(new PhotoGalleryPhoto(self.data.images[imagePosition]), true);

                self.configurePhotoEditOptions(imagePosition);

                callback();
            };

            image.src = this.data.images[imagePosition].url;
        };

        // Correct panning on thumbnails, used when the size of them changes
        module.correctThumbnails = function () {
            var self = this;

            // Disable automatic save from our image cropper
            this.activePhotoPosition = null;

            // Set our heights to false so we render the table naturally
            var oldHeights = [];
            for (var i = 0; i < this.data.images.length; i++) {
                oldHeights.push(this.data.images[i].height);
                this.data.images[i].height = false;
            }

            // Render so we can get our new thumbnail width
            self._reloadHtml();

            // Set our heights back to normal
            for (var i = 0; i < this.data.images.length; i++) {
                this.data.images[i].height = oldHeights[i];
            }

            var slides = this.getGridElements();

            // Check to make sure all images have been corrected and then render
            var renderView = function () {
                var overlayEllipsisFunc = _.bind(self._overlayEllipsis, self);
                overlayEllipsisFunc();
                self.$('.ellip-anchor').on('click', function (evt) {
                    evt.preventDefault();
                }); // prevent default click handler

                // remove events from parent from prior rendering, to prevent memory leaks (I hope)
                self.el.closest('.webs-bin').off('.photo_gallery');

                // add an event handler to the parent webs-bin for hover, as this changes the bg-color in edit mode
                self.el.closest('.webs-bin').on('mouseenter.photo_gallery', overlayEllipsisFunc);
                self.el.closest('.webs-bin').on('mouseleave.photo_gallery', overlayEllipsisFunc);
            };

            for (var i = 0; i < this.data.images.length; i++) {
                if ($(slides[i]).hasClass("selected")) {
                    var slideWrapper = $(slides[i]).find(".slide-wrapper")
                    break;
                }
            }

            if (self.el.find(".slide[data-_i_=0]").height()) {
                self.data.containerHeight = self.el.find(".slide[data-_i_=0]").height() + "px";
                // force the module to be dirty and force push the changes into mongo
                self.dirty();
            }

            // Adjust position of thumbs since the size changed
            for (var i = 0; i < this.data.images.length; i++) {

                var slide = $(slides[i]);
                var slideDimensions = {};
                var captionTitleHeight = slide.find('.webs-gallery-grid-title-caption-social-box').outerHeight(true) || 0;

                slideDimensions.width = slideWrapper.width();
                slideDimensions.height = slideWrapper.height() - captionTitleHeight;
                var thumbnailResizeFactor = slideDimensions.width / this.data.images[i].width;

                // Correct existing images
                if (this.data.images[i].width) {

                    // Correct position since we changed size
                    this.data.images[i].top = this.data.images[i].top * thumbnailResizeFactor;
                    this.data.images[i].left = this.data.images[i].left * thumbnailResizeFactor;

                    var oldSlideDimensions = {
                        width: this.data.images[i].width,
                        height: this.data.images[i].height
                    };
                    this.data.images[i] = _.extend(this.data.images[i], slideDimensions);

                    // Fetch our image dimensions
                    var image = new Image();

                    image.onload = (function (loop) {
                        return function () {

                            var oldImageWidthPixels = Math.round(oldSlideDimensions.width * self.data.images[loop].imageWidth / 100);
                            var oldImageHeightPixels = Math.round(oldImageWidthPixels * (this.height / this.width));
                            var missingVerticalSpace = (slideDimensions.height - (oldImageHeightPixels + self.data.images[loop].top));

                            // Check to see if we have enough image to change to square crop
                            if (thumbnailResizeFactor == 1 && slideDimensions.height != oldSlideDimensions.height && missingVerticalSpace > 0) {

                                // Detect if we can just shift our image down to get what we need
                                if ((self.data.images[loop].top * - 1) > missingVerticalSpace) {
                                    self.data.images[loop].top += missingVerticalSpace;
                                } else {

                                    // Zoom to get the extra image we need
                                    var newImageWidthPixels = (oldImageWidthPixels * slideDimensions.height) / oldImageHeightPixels;
                                    var zoom = newImageWidthPixels / slideDimensions.width;

                                    self.data.images[loop].top = 0;
                                    self.data.images[loop].imageWidth = (zoom * 100) + 1;
                                }

                                self.imageViews[loop].changeModel(new PhotoGalleryPhoto(self.data.images[loop]), true);
                            }

                            //remove the loading cover
                            self.imageViews[loop].removeLoadingCover();
                            self.configurePhotoEditOptions(loop);

                            renderView();
                        }
                    }(i));

                    image.src = this.data.images[i].url;

                } else {
                    this.configureNewImage(i, slideDimensions, function () {
                        renderView();
                    });
                }
            }

            // Force edit to update
            self._reloadHtml();
        };

        module.fit = function (width, persist) {
            this.correctThumbnails();
        };

        module.attachControls = function () {
            var mouseInButton = false;

            this.el.on('mouseenter', '.button', function (evt) {
                mouseInButton = true;
            });

            this.el.on('mouseleave', '.button', function (evt) {
                mouseInButton = false;
            });

            // Configure and show toolbar
            var photoGalleryType = this.data.type;

            // leave the possible to implement photo gallery into this module
            if (photoGalleryType == "grid") {
                this.toolbar = bldr.toolbar.create('photo_gallery');
            } else {
                this.toolbar = bldr.toolbar.create('photo_gallery_slideshow');
            }
            var toolbarModuleValues = {
                'images': this.data.images
            };

            _.extend(toolbarModuleValues, this.data.settings[photoGalleryType]);
            this.toolbar.show(toolbarModuleValues);

            if (!this.data.settings.grid.deckInspectah) {
                this.toolbar.find('.w-dropdown-container.w-tbui-hoverEffect').parent('li').hide();
                this.toolbar.find('.w-dropdown-container.w-tbui-hoverEffectColor').parent('li').hide();
            } else {
                this.toolbar.find('.w-dropdown-container.w-tbui-hoverEffectColor').parent('li')
                .toggle(this.data.settings.grid.hoverEffect !== 'hover-effect-none');
            }

            var self = this;
            this.toolbar.addListener({
                'images': function (data) {
                    self.data.images = data;
                    self.correctThumbnails();
                },
                'photosPerRow': function (val) {
                    self.data.settings.grid.photosPerRow = parseInt(val);
                    self.correctThumbnails();
                },
                'gridPadding': function (val) {
                    self.data.settings.grid.gridPadding = val;
                    self.correctThumbnails();
                },
                'cropStyle': function (val) {
                    self.data.settings.slideshow.cropStyle = val;
                    self.correctThumbnails();
                },
                'showTitle': function (val) {
                    self.data.settings.slideshow.showTitle = val;
                    self.correctThumbnails();
                },
                'deckInspectah': function (val) {
                    self.data.settings.grid.deckInspectah = val;
                    var hoverEffectLi = self.toolbar.find('.w-dropdown-container.w-tbui-hoverEffect').parent('li');
                    var hoverEffectColorLi = self.toolbar.find('.w-dropdown-container.w-tbui-hoverEffectColor').parent('li');

                    if (val) {
                        hoverEffectColorLi.toggle(self.data.settings.grid.hoverEffect !== 'hover-effect-none');
                        hoverEffectLi.show();
                    } else {
                        hoverEffectLi.hide();
                        hoverEffectColorLi.hide();
                    }
                },
                'hoverEffect': function (val) {
                    self.toolbar.find('.w-dropdown-container.w-tbui-hoverEffectColor').parent('li').toggle(val !== 'hover-effect-none');
                    self.data.settings.grid.hoverEffect = val;
                },
                'hoverEffectColor': function (val) {
                    self.data.settings.grid.hoverEffectColor = val;
                },
                'autoplay': function (val) {
                    self.data.settings.slideshow.autoplay = val;
                    self.correctThumbnails();
                },
                'duration': function (val) {
                    self.data.settings.slideshow.duration = val;
                    self.correctThumbnails();
                },
                'transition': function(val) {
                    self.data.settings.slideshow.transition = val;
                    self.correctThumbnails();
                }
            });

            this.el.bind("dblclick", function () {
                if (!mouseInButton) {
                    self.showSettingsDialog();
                }
            });
        };

        module._reloadHtml = function () {
            if (this.data.images && this.data.images[this.data.currentSlide]) {
                _.map(this.data.images, function(image) {
                    return image.selected = false;
                });
                this.data.images[this.data.currentSlide].selected = true;
            }

            this.el.html(this.html());
            this.imageViews = [];
            for (var i = 0; i < this.data.images.length; i++) {
                this.imageViews[i] = new PhotoGalleryImageView({
                    model: new PhotoGalleryPhoto(this.data.images[i]),
                    imageReady: false
                });
                this.imageViews[i].render();
                this.getGridElements(i).find('.image-container').append(this.imageViews[i].el);
            }

            if (this.data.type == "grid") {
                this.setupSortable();
            }
        }


    })(mod, ext);

    return ModuleClassLoader.register('photo_gallery_slideshow', mod, ext);
});




// webstore_products Module (custom)

define('module/webstore_products', ['jquery', 'internal/sitebuilder/common/ModuleClassLoader', 'translate!webs.module.webstore_products', 'underscore', 'backbone', 'translate!webs.module.webstore_products', 'site/modules/photo_gallery/SwappableSortableView', 'instance/site/appList'],
function($, ModuleClassLoader, translate, _, Backbone, translate, SwappableSortableView, appList) {

    var mod = {}, ext = {};
    (function(module, extend) {

        // Show Cover?
        extend.shouldShowCover = function() {
            return false;
        };

        // Is module movable?
        extend.isMovable = function() {
            return true;
        };

        // Is module removable?
        extend.isRemovable = function() {
            return true;
        };



        // Module HTML Template
        extend.template = '<div class="w-webstore_products product-list-container webs-text">\
        <div class="moveable"/>\
        \
        <div class="product-list">\
        <div class="row">\
        {{#each products}}\
        <div class="product products-per-row-{{../settings.productsPerRow}}" data-product-id="{{id}}">\
        </div>\
        {{{_every_ ../settings.productsPerRow "</div><div class=\'row\'>"}}}\
        {{/each}}\
        {{#unless products.length}}\
        <div class="bldr-placeholder-webstore-products bldr-placeholder-element {{#if webstoreInstalled}} webstore-installed {{else}} webstore-not-installed {{/if}}">\
        {{#if webstoreInstalled}}\
        <span class="placeholder-text">{{#l}}webs.module.webstore_products.edit.settings.noProduct{{/l}}</span>\
        {{else}}\
        <div class="placeholder-text">\
        <div>{{#l}}webs.module.webstore_products.edit.settings.webstoreNotInstalled{{/l}}</div>\
        <a class="fw-button" href="/s/app/install?section=app&action=install&app=webstore&next=hidden&title={{#l}}webs.module.webstore_products.edit.settings.webstoreTitle{{/l}}">\
        {{#l}}webs.module.webstore_products.edit.settings.installWebstore{{/l}}\
        </a>\
        </div>\
        {{/if}}\
        </div>\
        {{/unless}}\
        </div>\
        </div>\
        </div>\
        \
        <div class="placeholder-template">\
        <div class="product products-per-row-{{settings.productsPerRow}}">\
        <div class="product-image-container">\
        <a href="#" class="product-image-anchor">\
        </a>\
        </div>\
        </div>\
        </div>\
        \
        <script id="product-Template" type="text/template">\
        <div>\
        <div class="product-settings">\
        <a class="remove-product"></a>\
        </div>\
        \
        <div class="product-image-container">\
        <a class="product-image-anchor showProduct" href="<%= href %>" style="background-image: url(<%= imageUrl %>)"></a>\
        </div>\
        \
        <% if (settings.showName) { %>\
        <a class="product-name showProduct" href="<%= href %>"><p><%= name%></p></a>\
        <% } %>\
        \
        <% if (settings.showPrice) { %>\
        <div class="product-price"><p>\
        <% if (sale_price_display) { %>\
        <span class="sale-price-display theme-color-text"><%= sale_price_display %></span>\
        <span class="price-display"><%= price_display %></span>\
        <% } else { %>\
        <span class="theme-color-text"><%= price_display %></span>\
        <% } %>\
        </p></div>\
        <% } %>\
        \
        <% if (settings.showProductBtn) { %>\
        <div class="product-view-btn">\
        <div class="w-button-container w-button-<%= theme_slug %>_default w-button-default">\
        <div class="w-button-align">\
        <a class="w-button showProduct" href="<%= href %>">\
        <div class="w-button-inner">\
        <div class="w-button-text-wrapper">\
        <div class="w-button-text"></div>\
        </div>\
        </div>\
        </a>\
        </div>\
        </div>\
        </div>\
        <% } %>\
        </div>\
        </script>\
        ';

        // Module Default Data
        extend.defaultData = {
            "products": [],
            "settings": {
                "productsPerRow": 3,
                "showName": true,
                "showPrice": true,
                "showProductBtn": true
            }
        };

        // SubModules

        // Module Styles
        extend.styles = {
            "base": {
                "global": {
                    "css": "view.less"
                },
                "slug": "base"
            }
        };
        if (!extend.styles['base']['global']) {
            extend.styles['base']['global'] = {};
        }
        extend.styles['base']['global']['js'] = null;
        extend.defaultStyle = extend.styles['base'];

        // Register Toolbars
        bldr.toolbar.register('webstore_products', {
            "icon": "webstore_products",
            "items": [{
                "label": "webstore_products.manageProducts",
                "name": "products",
                "dialog": {
                    "url": "/s/modules/webstore_products/v1.13/webstore_products.dialog.html",
                    "heading": "Products Settings",
                    "localizeHeading": "webs.module.webstore_products.dialog.header",
                    "width": 750,
                    "height": 500
                },
                "style": "blue"
            }, {
                "_comment": "Should be refactored to use webs.util.Settings and modify localizeToolbarText() in toolbar.js",
                "type": "menu",
                "label": "webstore_products.settings",
                "menuLabel": "Product Settings",
                "menu": [{
                    "type": "dropdown",
                    "label": "webstore_products.productsPerRow",
                    "name": "productsPerRow",
                    "width": 50,
                    "display": "inline",
                    "options": [{
                        "label": 1,
                        "value": 1
                    }, {
                        "label": 2,
                        "value": 2
                    }, {
                        "label": 3,
                        "value": 3
                    }, {
                        "label": 4,
                        "value": 4
                    }
                    ]
                }, {
                    "type": "switch-bare",
                    "label": "webstore_products.name",
                    "name": "showName",
                    "display": "inline"
                }, {
                    "type": "switch-bare",
                    "label": "webstore_products.price",
                    "name": "showPrice",
                    "display": "inline"
                }, {
                    "type": "switch-bare",
                    "label": "webstore_products.viewProductBtn",
                    "name": "showProductBtn",
                    "display": "inline"
                }
                ]
            }
            ]
        });

        // View JS
        var ProductView = Backbone.View.extend({

            events: {
                'click .showProduct': 'showProduct'
            },

            initialize: function (options) {
                this.opts = options || {};
                this.model.on('change', this.modelChanged, this);
                this.template = _.template(this.opts.template);
            },

            render: function () {
                var data = $.extend({}, this.model.attributes, {
                    'settings': this.opts.settings,
                    'href': '/apps/webstore' + this.model.get('url')
                });
                this.$el.html(this.template(data));
                this.$('.product-view-btn a .w-button-text').text(translate('webs.module.webstore_products.edit.viewProductBtn'));
                this.$('remove-product').attr('title', translate('webs.module.webstore_products.edit.settings.removeProduct'));
            },

            modelChanged: function () {
                this.updateUI();
            },

            updateUI: function (options) {
                this.opts = _.extend(this.opts, options);
                this.model.set('imageUrl', this.getImageProcessorUrl(this.model.get('image_id')), {
                    silent: true
                });
                this.render();
            },

            showProduct: function (evt) {
                if (this.opts.blockNavigation) {
                    evt.preventDefault();
                }
            },

            getImageProcessorUrl: function (imageId) {
                if (imageId && imageId.toString().indexOf('|') > - 1) {
                    imageId = imageId.split('|')[0];
                }

                var imageHeight = this.$('.product-image-anchor').height();
                var imageWidth = this.$('.product-image-anchor').width();
                var imageProcessorURL = webs.props.imageProcessorServer + '/fit/{width}x{height}/';
                if (this.model.get('image_crop_style') == 'square') {
                    imageProcessorURL = webs.props.imageProcessorServer + '/square/{width}/';
                }
                var thumbsViewURL = webs.props.thumbServer + '/Members/viewThumb.jsp';

                var imageUrl = (imageProcessorURL).replace(/{width}/, imageWidth).replace(/{height}/, imageHeight)
                + thumbsViewURL + encodeURIComponent('?fileID={imageId}&size=full'.replace(/{imageId}/, imageId));

                return imageUrl;
            }
        });

        var Product = Backbone.Model.extend({
            idAttribute: 'id',
            defaults: {
                'imageUrl': webs.props.staticServer + '/static/projects/finch/images/loader_animation_lightbox_grey.gif',
                'name': "",
                'price': "",
                'price_display': "",
                'sale_price': '',
                'sale_price_display': '',
                'url': "",
                'image_crop_style': 'fit',
                'theme_slug': (webs && webs.theme) ? webs.theme.slug : ''
            }
        });

        var ProductCollections = Backbone.Collection.extend({
            model: Product,
            url: '/manageapp/webstore/products.json'
        });

        $.extend(module, {
            oneLoaded: function () {
                this.rebuildModule(false, true);
            },

            rebuildModule: function (blockNavigation, showingView) {
                this.productsCollection = new ProductCollections(this.data.products);
                if (showingView) {
                    this.productsCollection.url = '/apps/webstore/products.json';
                }

                this.createProductsViews(blockNavigation);
                this.productsCollection.fetch({
                    merge: true,
                    add: true
                });
            },


            createProductsViews: function (blockNavigation) {
                var self = this;
                this.productsViewList = [];
                var html = self.el.find('#product-Template').html();
                _.each(this.data.products, function (product) {
                    var productModel = self.productsCollection.get(product.id);
                    var productView = new ProductView({
                        el: self.el.find('div[data-product-id=' + product.id + ']'),
                        model: productModel,
                        settings: self.data.settings,
                        blockNavigation: blockNavigation,
                        template: self.el.find('#product-Template').html()
                    });
                    productView.render();
                    self.productsViewList.push(productView);
                });
            }

        });


        // Edit JS
        /**
         * Initialize edit mode
         */

        module.safeOneLoaded = function () {
            this.bind("rendered", _.once(_.bind(this.initProducts, this)));
        };

        module.initProducts = function () {
            var self = this;

            this.viewData = {
                webstoreInstalled: true
            };

            this.bind("rendered", function () {
                this.createProductsViews(true)
            }, this);

            if (this.data.products.length == 0) {
                this.el.find('.product-list-container').addClass('loadingSpinner');
                this.el.find('.product-list').hide();
                appList.fetch({
                    success: function () {
                        self.el.find('.product-list-container').removeClass('loadingSpinner');
                        self.el.find('.product-list').show();

                        self.viewData.webstoreInstalled = appList.isAppInstalled("webstore");

                        self.el.html(self.html());
                    }
                });
            } else {
                self.el.html(self.html());
                self.rebuildModule(true);
            }

            this.setupSettingsForProduct();
        };

        module.showSettingsDialog = function () {
            this.toolbar.click("products");
        };

        module.edit = function () {};

        module.onActivate = function () {
            this.activated = true;
            this.setupSortable();
        };

        module.onDeactivate = function () {
            if (this.toolbar) {
                this.toolbar.reset();
            }
            this.activated = false;
            if (this.sortableView) {
                this.sortableView.detach();
                this.sortableView = null;
            }
        };

        module.fit = function (width, persist) {};

        module.attachControls = function () {
            if (!this.viewData.webstoreInstalled) {
                return;
            }

            // Configure and show toolbar
            this.toolbar = bldr.toolbar.create('webstore_products');
            this.toolbar.show(this.data.settings);

            var self = this;
            this.toolbar.addListener({
                'products': function (data) {
                    // In order to keep current products and not have duplicates (Backbone collection does not allow duplicate ids)
                    var productsCollection = new Backbone.Collection($.merge(self.data.products, data));
                    self.data.products = productsCollection.toJSON();
                    self.el.html(self.html());
                    self.rebuildModule(true);
                    self.setupSortable();
                },
                'productsPerRow': function (val) {
                    self.data.settings.productsPerRow = parseInt(val);
                    self.el.html(self.html());
                    self.createProductsViews(true);
                    self.updateProductsViews();
                    self.setupSortable();
                },
                'showName': function (val) {
                    self.data.settings.showName = val;
                    self.updateProductsViews();
                },
                'showPrice': function (val) {
                    self.data.settings.showPrice = val;
                    self.updateProductsViews();
                },
                'showProductBtn': function (val) {
                    self.data.settings.showProductBtn = val;
                    self.updateProductsViews();
                }
            });

            this.el.bind("dblclick", function () {
                self.showSettingsDialog();
            });
        };

        module.updateProductsViews = function () {
            var settings = {
                'settings': this.data.settings
            };
            _.each(this.productsViewList, function (view) {
                view.updateUI(settings);
            })
        };

        module.setupSettingsForProduct = function () {
            var self = this;
            this.el.on('click', '.w-webstore_products .product-list .product .remove-product', function () {
                var productId = $(this).parents('.product').data('product-id');
                var newProductsOrderArray = [];
                _.each(self.data.products, function (productInfo) {
                    if (productInfo.id != productId) {
                        newProductsOrderArray.push(productInfo)
                    }
                });
                self.data.products = newProductsOrderArray;
                self.el.html(self.html());
                self.createProductsViews(true);
                self.setupSortable();
            });
        };


        // Setting up reorder for products
        module.setupSortable = function () {
            if (this.sortableView) {
                this.sortableView.detach();
                this.sortableView = null;
            }


            if (this.activated && this.data.products.length > 1) {
                var self = this;

                // Update moveable div to have correct height & width so it has same dimensions as thing we are trying to move
                var $moveableContainer = this.el.find('.moveable');
                var $productContainer = this.el.find('.product');
                $moveableContainer.css({
                    width: $productContainer.width(),
                    height: $productContainer.height()
                });

                var $placeholder = $(this.el.find('.placeholder-template').html());


                this.sortableView = new SwappableSortableView({
                    el: $(this.el),
                    $placeholder: $placeholder,
                    $moveableContainer: $moveableContainer,
                    $scrollDiv: self.getScrollableElement(),
                    nonDragSelector: '',
                    elementSelector: '.w-webstore_products .product',
                    container: this.el.find('.w-webstore_products .product-list'),
                    scrollTarget: 100,
                    scrollSpacingTop: 50,
                    scrollSpacingBottom: 200
                });

                this.sortableView.off('elementReordered');
                this.sortableView.on('elementReordered', function () {
                    var newProductsOrderArray = [];
                    self.el.find('.w-webstore_products .product-list .product').each(function (index, productDiv) {
                        var productId = $(productDiv).data('product-id');
                        var product = _.find(self.data.products, function (productInfo) {
                            return productInfo.id == productId;
                        });
                        newProductsOrderArray.push(product);
                    });
                    self.data.products = newProductsOrderArray;
                });

                this.sortableView.start();
            }
        };


        // Get Scroll div (html or body) so when reorder we can scroll
        module.getScrollableElement = function () {
            if (!this.$scrollDiv) {
                var self = this;
                this.$scrollDiv = this.el.parents('body');
                $.each([this.el.parents('body'), this.el.parents('html')], function (index, value) {
                    value.scrollTop(value.scrollTop() + 1);
                    if (value.scrollTop() > 0) {
                        value.scrollTop(value.scrollTop() - 1);
                        self.$scrollDiv = value;
                        return false;
                    }
                });
            }
            return this.$scrollDiv;
        };

    })(mod, ext);

    return ModuleClassLoader.register('webstore_products', mod, ext);
});





