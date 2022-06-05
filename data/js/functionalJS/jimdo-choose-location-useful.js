try{

        if(/\/templates\/?(?:#|\?|$)/.test(location.pathname)) {
                cxApi.setDomainName('www.jimdo.com');
                cxApi.setAllowHash(false);
                var chosenVariation = cxApi.chooseVariation();
                if(chosenVariation === 1){
                        window.showDirectSale = true;
                }

        }
        

}catch(e){}
