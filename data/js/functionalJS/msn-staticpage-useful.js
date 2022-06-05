require(["headData", "pageInstance", "mediator"], function (headData, pageInstance, mediator)
                {
                    var url = 'http://c.msn.com/c.gif?udc=true&rid=b655bdc7ca854bfd9ccc189a65e3e437&rnd=635530218979042579&rf=&tp=http%253A%252F%252Fwww.msn.com%252Fen-au%252F&di=812&lng=en-au&cv.product=prime&d.dg1=&d.dg2=&d.dg3=&d.dg4=&d.dgk=tmx.pc.webkit.safari.altnth&d.imd=0&d.b=Safari&d.bv=7.1&d.p=Unknown&d.pv=Unknown%20Macintosh';
                    if (headData && headData.clientSettings && headData.clientSettings.static_page)
                    {
                        mediator.pub(pageInstance.eventName);
                        var rid = pageInstance.getActivityId();
                        url = url.replace(/([?&]rid=)[^&#]*/i, "$1" + rid);
                    }
                    var img = new Image();
                    img.onload = img.onerror = function ()
                    {
                        img.onload = img.onerror = null;
                    };
                    img.src = url;
                });