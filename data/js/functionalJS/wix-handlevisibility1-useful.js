
        if (document.webkitVisibilityState)
        {
            if (document.webkitVisibilityState == "prerender")
            {
                var frog_img = new Image();
                frog_img.src = "http://frog.wix.com/pblc?evid=1";
                document.addEventListener("webkitvisibilitychange",handleVisibilityChange,false);
            }
        }
        if (isMobile)
        {
            social_plugs.setAttribute('mb','true');
            function sociel_mb_nav(e)
            {
                for (var i = 0; i < social_plugs.children.length; i++)
                {
                    if (e.currentTarget == social_plugs.children[i])
                    {
                        if (e.currentTarget.getAttribute('down') == 'true')
                        {
                            social_plugs.children[i].setAttribute('down','false');
                        } else
                        {
                            social_plugs.children[i].setAttribute('down','true');
                        }
                    } else
                    {
                        social_plugs.children[i].setAttribute('down','false');
                    }
                }
            }
            fb_item.addEventListener('click',sociel_mb_nav,false);
            tw_item.addEventListener('click',sociel_mb_nav,false);
            pin_item.addEventListener('click',sociel_mb_nav,false);
        }
    