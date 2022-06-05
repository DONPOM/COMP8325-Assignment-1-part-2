
          $(document).ready(function () {

              NetdaniaConverter.CurrencyConverter({ containerID: 'convertor' });

          var printvindue;

          aabenvindue = function() {
              printvindue = window.open('/UI/printside.html', 'print', 'width=850,height=620,scrollbars=yes,resizable=yes')
          }
                  
          function pageLoad(sender, args) {
              if (!args.get_isPartialLoad()) {

                  try {
                      var background = $find("programmaticModalPopupBehavior")._backgroundElement;
                      if (background != null) {
                          background.onclick = function () { hideModalPopupViaClient(); }
                      }
                      document.onkeydown = function () { hideModalPopupViaClient(); }
                  } catch (ex) { }
              }
          }

          var redirect = false;
          var redirectToNetDania = false;



          ddmegamenu.docinit({
              menuid: 'solidmenu'
              //dur:200 //<--no comma after last setting
          })
       });      
    