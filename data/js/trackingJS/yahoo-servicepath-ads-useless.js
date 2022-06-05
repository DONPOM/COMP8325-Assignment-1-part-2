
        var w = window, D = w.DARLA, C = {"useYAC":0,"servicePath":"https:\/\/au.yahoo.com\/sdarla\/php\/fc.php","xservicePath":"","beaconPath":"https:\/\/au.yahoo.com\/sdarla\/php\/b.php","renderPath":"","allowFiF":false,"srenderPath":"https:\/\/s.yimg.com\/rq\/darla\/2-8-4\/html\/r-sf.html","renderFile":"https:\/\/s.yimg.com\/rq\/darla\/2-8-4\/html\/r-sf.html","sfbrenderPath":"https:\/\/s.yimg.com\/rq\/darla\/2-8-4\/html\/r-sf.html","msgPath":"https:\/\/au.yahoo.com\/sdarla\/2-8-4\/html\/msg.html","cscPath":"https:\/\/s.yimg.com\/rq\/darla\/2-8-4\/html\/r-csc.html","root":"sdarla","edgeRoot":"http:\/\/l.yimg.com\/rq\/darla\/2-8-4","sedgeRoot":"https:\/\/s.yimg.com\/rq\/darla\/2-8-4","version":"2-8-4","tpbURI":"","hostFile":"https:\/\/s.yimg.com\/rq\/darla\/2-8-4\/js\/g-r-min.js","beaconsDisabled":true,"rotationTimingDisabled":true,"fdb_locale":"What don't you like about this ad?|<span>Thank you<\/span> for helping us improve your Yahoo experience|I don't like this ad|I don't like the advertiser|It's offensive|Other (tell us more)|Send|Done","positions":{"RSCH":{"w":1,"h":1},"FOOT9":{"w":1,"h":1},"LREC":{"w":1,"h":1},"MAST":[],"FPAD":{"w":300,"h":250},"TL1":{"w":120,"h":45},"FPAD2":{"w":300,"h":250}},"lang":"en-AU"};
        if (D && C) {
            C.positions = {"FOOT9":{"clean":"my-adsFOOT9","dest":"my-adsFOOT9-iframe","w":1,"h":1},"MAST":{"pos":"MAST","clean":"my-adsMAST","dest":"my-adsMAST-iframe","fr":"expIfr_exp","w":970,"h":15,"metaSize":1,"supports":{"exp-ovr":1,"exp-push":1,"lyr":1}},"FPAD":{"clean":"my-adsFPAD","dest":"my-adsFPAD-iframe","metaSize":1,"supports":{"exp-ovr":1,"lyr":1}},"TL1":{"clean":"my-adsTL1","dest":"my-adsTL1-iframe","w":128,"h":134},"FPAD2":{"clean":"my-adsFPAD2","dest":"my-adsFPAD2-iframe","metaSize":1,"supports":{"exp-ovr":1,"lyr":1}}};
            C.k2E2ERate=50;
C.k2Rate=1;
C.viewRate=100;

            C.events = null;
               
            C.onFinishPosRender = function(posId, reqList, posItem) {
                var curAd = document.getElementById("my-ads"+posId),
                    posSize = (posItem && posItem.size);
                if (curAd && posSize && posSize != "1x1") {
                    curAd.className = curAd.className.replace('D-n', '');
                } else {
                    curAd.className = curAd.className.replace('Ht-250', '');
                }
                
            };
                
            //call back when the ad is expanded or collapsed
            C.onPosMsg = function (msg_name, data)  {
                var visible;
                if(msg_name == "collapse" && data == "MAST") {
                    var bodyTag = document.getElementsByTagName("body")[0];
                    bodyTag.className = bodyTag.className.replace('mastAdExpanded', ''); 
                    bodyTag.className += " " +  "mastAdCollapsed";
                }
                if(msg_name == "exp-push" && data == "MAST") {
                    var bodyTag = document.getElementsByTagName("body")[0];
                    bodyTag.className = bodyTag.className.replace('mastAdCollapsed', '');
                    bodyTag.className += " " +  "mastAdExpanded";
                }
                
                /* generic ad expansion logic */
                if(msg_name == "collapse") {
                    var bodyTag = document.getElementsByTagName("body")[0];
                    bodyTag.className = bodyTag.className.replace(data + "-ad-expanded", '');
                }
                
                if(msg_name == "exp-ovr") {
                    var bodyTag = document.getElementsByTagName("body")[0];
                    bodyTag.className += " " + data + "-ad-expanded";
                }
                
                if (msg_name === 'geom-update') {
                    visible = D.render.RenderMgr.get(data).viewedAt > 0;
                    // geom-update event will always be available when Y is available
                    if (YMedia && visible) {
                        YMedia.Global.fire('ads:beacon', {id: data});
                    }
                }
            };
                        
            if ("OK" == D.config(C)) {
                setTimeout(function(){
                D.render();
                }, 200);
            }
        }
    