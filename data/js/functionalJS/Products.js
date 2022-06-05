// JScript File
// Provider|Code|text|preloading|inSearch(ignored)|doGrouping|isExpanded

function RegisterIE() {}

function RegisterInstruments(product) {
        if (product != 'NetStation') {
		    document.writeln('<param name="iexplore_size" value="400|450" />');
		    document.writeln('<param name="iexplore_preload_instruments" value="false" />');
		}
		
		document.writeln('<param name="iexplore_default_grouping" value="1"/>');
		
//Indices

		document.writeln('<param name="iexplore_market_1" value="@HTML|aust_|Australia|false|true|false|false" />');
        document.writeln('<param name="aust_1" value="I*XJO.as|idc_rtb;S&P/ASX 200;Valuta;0" />');
        document.writeln('<param name="iexplore_market_1_path" value="Indices" />');
	    
	    document.writeln('<param name="iexplore_market_2" value="@HTML|bel_|Belgium|false|true|false|false" />');
        document.writeln('<param name="bel_1" value="!BEL20/BE|idc_dlb;Belgium 20;Valuta;0" />');
        document.writeln('<param name="bel_2" value="BELS.en|idc_dlb;BEL Small;Valuta;0" />');
        document.writeln('<param name="bel_3" value="BELM.en|idc_dlb;BEL Mid;Valuta;0" />');
        document.writeln('<param name="iexplore_market_2_path" value="Indices" />');
		
		document.writeln('<param name="iexplore_market_3" value="@HTML|can_|Canada|false|true|false|false" />');
        document.writeln('<param name="can_1" value="I*0000.tt|idc_dlc;S&P/TSX Composite Index;Valuta;0" />');
        document.writeln('<param name="can_2" value="I*TX60.tt|idc_dlc;S&P/TSX 60 Index;Valuta;0" />');
        document.writeln('<param name="iexplore_market_3_path" value="Indices" />');
		
		document.writeln('<param name="iexplore_market_4" value="@HTML|sha_|China|false|true|false|false" />');
        document.writeln('<param name="sha_1" value="I*000001.sh|idc_dla;SSE Composite Index;Valuta;0" />');
        document.writeln('<param name="iexplore_market_4_path" value="Indices" />');
  
        document.writeln('<param name="iexplore_market_5" value="@HTML|den_|Main Indices|false|true|false|false" />');
        document.writeln('<param name="den_1" value="OMXC20.co|idc_dla;OMXC20;Valuta;0" />');
		document.writeln('<param name="den_2" value="OMXC20CAP.co|idc_dla;OMXC20CAP;Valuta;0" />');
        document.writeln('<param name="iexplore_market_5_path" value="Indices|Denmark" />');

        document.writeln('<param name="iexplore_market_6" value="idc_dla|DENMARK|All Indices|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_6_types" value="4" />');
        document.writeln('<param name="iexplore_market_6_path" value="Indices|Denmark" />');       
 
		document.writeln('<param name="iexplore_market_7" value="@HTML|ext_|Euronext|false|true|false|false" />');
		document.writeln('<param name="ext_1" value="N100.en|idc_dlb;Euronext 100;true;0" />');
		document.writeln('<param name="ext_2" value="EFC1.en|idc_dlb;FTSEurofirst 100;true;0" />');
		document.writeln('<param name="ext_3" value="EF80.en|idc_dlb;FTSEurofirst 80;true;0" />');
		document.writeln('<param name="ext_4" value="N150.en|idc_dlb;NEXT 150;true;0" />');
        document.writeln('<param name="ext_5" value="CN20.en|idc_dlb;NEXT CAC 20;true;0" />');
		document.writeln('<param name="iexplore_market_7_path" value="Indices" />');
		
        document.writeln('<param name="iexplore_market_8" value="@HTML|finidx_|Main Index|false|true|false|false" />');
        document.writeln('<param name="finidx_1" value="OMXH25.he|idc_dla;OMXH25;Valuta;0" />');
        document.writeln('<param name="iexplore_market_8_path" value="Indices|Finland" />');

        document.writeln('<param name="iexplore_market_9" value="idc_dla|FINLAND|All Indices|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_9_path" value="Indices|Finland" />');
        document.writeln('<param name="iexplore_market_9_types" value="4"/>');
	    document.writeln('<param name="iexplore_market_9_regex" value="1|(^([^0-9]))"/>');

        document.writeln('<param name="iexplore_market_10" value="@HTML|fra_|France|false|true|false|false" />');
        document.writeln('<param name="fra_1" value="PX1.en|idc_dlb;CAC40;Valuta;0" />');
        document.writeln('<param name="fra_2" value="CACMD.en|idc_dlb;CAC MID 60;Valuta;0" />');
        document.writeln('<param name="fra_3" value="CACS.en|idc_dlb;CAC SMALL 90;Valuta;0" />');
        document.writeln('<param name="fra_4" value="CACMS.en|idc_dlb;CAC Mid&Small 190;Valuta;0" />');
        document.writeln('<param name="iexplore_market_10_path" value="Indices" />');
		
		document.writeln('<param name="iexplore_market_11" value="@HTML|gidx_|Germany|false|true|false|false" />');
		document.writeln('<param name="gidx_1" value="DAX.de|idc_dla;DAX;true;0" />');
		document.writeln('<param name="gidx_2" value="MDAX.de|idc_dla;MDAX;true;0" />');
		document.writeln('<param name="gidx_3" value="SDXP.de|idc_dla;SDAX;true;0" />');
		document.writeln('<param name="gidx_4" value="TDXP.de|idc_dla;TecDAX;true;0" />');
		document.writeln('<param name="iexplore_market_11_path" value="Indices" />');
				
        document.writeln('<param name="iexplore_market_12" value="@HTML|indi_|Main Index|false|true|false|false" />');
		document.writeln('<param name="indi_1" value="I*NIFTY.in|idc_dlc;Nifty 50;true;0" />');
        document.writeln('<param name="iexplore_market_12_path" value="Indices|India" />');

        document.writeln('<param name="iexplore_market_13" value="idc_dlc|XNSE|All Indices|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_13_path" value="Indices|India" />');
        document.writeln('<param name="iexplore_market_13_types" value="4"/>');
//		document.writeln('<param name="iexplore_market_13_regex" value="1|(^([^0-9]))"/>');

		document.writeln('<param name="iexplore_market_14" value="@HTML|jidx_|Japan|false|true|false|false" />');
		document.writeln('<param name="jidx_1" value="N225|idc_dla;Nikkei 225;true;0" />');
		document.writeln('<param name="iexplore_market_14_path" value="Indices" />');
			
		document.writeln('<param name="iexplore_market_15" value="@HTML|net_|Netherlands|false|true|false|false" />');
		document.writeln('<param name="net_1" value="AEX.en|idc_dlb;Amsterdam EX;true;0" />');
		document.writeln('<param name="net_2" value="AMX.en|idc_dlb;AMX;true;0" />');
		document.writeln('<param name="net_3" value="ASCX.en|idc_dlb;AMS SMALL CAP INDX;true;0" />');
		document.writeln('<param name="iexplore_market_15_path" value="Indices" />');

		document.writeln('<param name="iexplore_market_16" value="@HTML|nor_|Norway|false|true|false|false" />');
		document.writeln('<param name="nor_1" value="OMXO20.st|idc_dla;OMXO20;true;0" />');
		document.writeln('<param name="iexplore_market_16_path" value="Indices" />');		
		
		document.writeln('<param name="iexplore_market_17" value="@HTML|por_|Portugal|false|true|false|false" />');
		document.writeln('<param name="por_1" value="PSI20.en|idc_dlb;PSI 20;true;0" />');
		document.writeln('<param name="por_2" value="BVLGR.en|idc_dlb;PSI General;true;0" />');
		document.writeln('<param name="iexplore_market_17_path" value="Indices" />');
		
		document.writeln('<param name="iexplore_market_18" value="@HTML|spa_|Spain|false|true|false|false" />');
		document.writeln('<param name="spa_1" value="IBEX.es|idc_rtb;IBEX 35;true;0" />');
		document.writeln('<param name="iexplore_market_18_path" value="Indices" />');
		
		document.writeln('<param name="iexplore_market_19" value="@HTML|stoidx_|Main Index|false|true|false|false" />');
        document.writeln('<param name="stoidx_1" value="OMXS30.st|idc_dla;OMXS30;Valuta;0" />');
        document.writeln('<param name="iexplore_market_19_path" value="Indices|Sweden" />');
   	
        document.writeln('<param name="iexplore_market_20" value="idc_dla|SWEDEN|All Indices|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_20_path" value="Indices|Sweden"/>');
        document.writeln('<param name="iexplore_market_20_types" value="4"/>');
//	    document.writeln('<param name="iexplore_market_20_regex" value="1|^([^0-9ABCDEFGHIJKLNSUXZ]|A[^tu]|B[^emz]|C[^ahnzy]|D[^ek]|E[^es]|F[^irR]|G[^bgir]|H[^ku]|I[^elmts]|J[^ep]|K[^ry]|L[^ru]|N[^lo]|S[^ge]|U[^s]|X[^s]|Z[^a]).*"/>');

		document.writeln('<param name="iexplore_market_21" value="@HTML|sw_|Swiss|false|true|false|false" />');
        document.writeln('<param name="sw_1" value="SMI.sw|idc_dla;SMI 20;Valuta;0" />');
        document.writeln('<param name="iexplore_market_21_path" value="Indices" />');
		
		document.writeln('<param name="iexplore_market_22" value="@HTML|tur_|Main Index|false|true|false|false" />');
        document.writeln('<param name="tur_1" value="XU100|idc_dla;ISE National 100;Valuta;0" />');
        document.writeln('<param name="iexplore_market_22_path" value="Indices|Turkey" />');
   	
        document.writeln('<param name="iexplore_market_23" value="idc_dla|XIST|All Indices|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_23_path" value="Indices|Turkey"/>');
        document.writeln('<param name="iexplore_market_23_types" value="4"/>');

		document.writeln('<param name="iexplore_market_24" value="@HTML|lsei_|United Kingdom|false|true|false|false" />');
		document.writeln('<param name="lsei_1" value="UKX.ln|idc_dla;FTSE 100;true;0" />');
		document.writeln('<param name="lsei_2" value="MCX|idc_dla;FTSE 250;true;0" />');
		document.writeln('<param name="lsei_3" value="NMX|idc_dla;FTSE 350;true;0" />');
		document.writeln('<param name="iexplore_market_24_path" value="Indices" />');
		
		document.writeln('<param name="iexplore_market_25" value="@HTML|uidx_|USA|false|true|false|false" />');
		document.writeln('<param name="uidx_1" value="!DJI|idc_rtb;Dow Jones Industrial 30;true;0" />');
		document.writeln('<param name="uidx_2" value="!DJC|idc_rtb;Dow Jones Composite;true;0" />');
		document.writeln('<param name="uidx_3" value="!DJT|idc_rtb;Dow Jones Transportation;true;0" />');
		document.writeln('<param name="uidx_4" value="!DJU|idc_rtb;Dow Jones Utilities;true;0" />');
		//document.writeln('<param name="uidx_5" value="NDX|idc_dla;Nasdaq 100;true;0" />');
		//document.writeln('<param name="uidx_6" value="@CCO|idc_dlb;Nasdaq Composite;true;0" />');
		document.writeln('<param name="iexplore_market_25_path" value="Indices" />');
		
	
//Shares Austria Real-time

        document.writeln('<param name="iexplore_market_26_path" value="Shares|Austria - (real-time)"  />');
        document.writeln('<param name="iexplore_market_26" value="idc_rtb|CHIXV#ATX.at|ATX|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_26_types" value="1"  />');

		document.writeln('<param name="iexplore_market_27_path" value="Shares|Austria - (real-time)"  />');
        document.writeln('<param name="iexplore_market_name_27" value="CHIXV;Austria">');
        document.writeln('<param name="iexplore_market_27" value="idc_rtb|CHIXV|All|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_27_types" value="1"  />');
        document.writeln('<param name="iexplore_market_27_regex" value="0|^[^0-9]*$"  />');
        document.writeln('<param name="iexplore_market_27_group_by_letter" value="true;20"  />');


//Shares Belgium Real-time
	
		document.writeln('<param name="iexplore_market_28_path" value="Shares|Belgium - (real-time)" />');
        document.writeln('<param name="iexplore_market_28" value="idc_rtb|CHIXB#BEL20|BEL 20|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_28_types" value="1" />');
       
        document.writeln('<param name="iexplore_market_29_path" value="Shares|Belgium - (real-time)"/>');
        document.writeln('<param name="iexplore_market_29" value="idc_rtb|CHIXB#BELM.en|BEL Mid|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_29_types" value="1" />');

        document.writeln('<param name="iexplore_market_30_path" value="Shares|Belgium - (real-time)"/>');
        document.writeln('<param name="iexplore_market_30" value="idc_rtb|CHIXB#BELS.en|BEL Small|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_30_types" value="1" />');

        document.writeln('<param name="iexplore_market_31_path" value="Shares|Belgium - (real-time)"/>');
        document.writeln('<param name="iexplore_market_name_31" value="CHIXB;Belgium">');
        document.writeln('<param name="iexplore_market_31" value="idc_rtb|CHIXB|All|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_31_types" value="1" />');
        document.writeln('<param name="iexplore_market_31_group_by_letter" value="true;20" />');
//	    document.writeln('<param name="iexplore_market_31_regex" value="0|^[^0-9]*$" />');
		
//Shares Belgium Delayed	    
   
 		document.writeln('<param name="iexplore_market_32_path" value="Shares|Belgium - (delayed) " />');
        document.writeln('<param name="iexplore_market_32" value="idc_dlb|XBRU#BEL20|BEL 20|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_32_types" value="1" />');
        
        document.writeln('<param name="iexplore_market_33_path" value="Shares|Belgium - (delayed) "/>');
        document.writeln('<param name="iexplore_market_33" value="idc_dlb|XBRU#BELM.en|BEL Mid|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_33_types" value="1" />');

        document.writeln('<param name="iexplore_market_34_path" value="Shares|Belgium - (delayed) "/>');
        document.writeln('<param name="iexplore_market_34" value="idc_dlb|XBRU#BELS.en|BEL Small|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_34_types" value="1" />');

        document.writeln('<param name="iexplore_market_35_path" value="Shares|Belgium - (delayed) "/>');
        document.writeln('<param name="iexplore_market_name_35" value="XBRU;Belgium">');
        document.writeln('<param name="iexplore_market_35" value="idc_dlb|XBRU|All|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_35_types" value="1" />');
        document.writeln('<param name="iexplore_market_35_group_by_letter" value="true;50" />');
	    document.writeln('<param name="iexplore_market_35_regex" value="0|^[^0-9]*$" />');
          
		  
//Shares Denmark Delayed

        document.writeln('<param name="iexplore_market_36_path" value="Shares|Canada - (delayed)"/>');
        document.writeln('<param name="iexplore_market_36" value="idc_dlc|XTSE#TXCOMP.tt|S&P/TSX Comp Index|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_36_types" value="1"/>');

        document.writeln('<param name="iexplore_market_37_path" value="Shares|Canada - (delayed)"/>');
        document.writeln('<param name="iexplore_market_37" value="idc_dlc|XTSE|All|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_37_types" value="1"/>');
		document.writeln('<param name="iexplore_market_37_group_by_letter" value="true;20" />');
		document.writeln('<param name="iexplore_market_37_regex" value="0|^[^0-9]*$" />');
		
//Shares Denmark Real-time

        document.writeln('<param name="iexplore_market_38_path" value="Shares|Denmark - (real-time)"/>');
        document.writeln('<param name="iexplore_market_38" value="idc_rtb|CHIXC#OMXC20|OMXC20|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_38_types" value="1"/>');
 
        document.writeln('<param name="iexplore_market_39_path" value="Shares|Denmark - (real-time)"/>');
        document.writeln('<param name="iexplore_market_39" value="idc_rtb|CHIXC|All|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_39_types" value="1"/>');
		document.writeln('<param name="iexplore_market_39_group_by_letter" value="true;20" />');
		document.writeln('<param name="iexplore_market_39_regex" value="0|^[^0-9]*$" />');
		
//Shares Denmark Delayed
 
        document.writeln('<param name="iexplore_market_40_path" value="Shares|Denmark - (delayed) "/>');
        document.writeln('<param name="iexplore_market_40" value="idc_dla|XCSE#OMXC20|OMXC20|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_40_types" value="1"/>');

        document.writeln('<param name="iexplore_market_41_path" value="Shares|Denmark - (delayed) "/>');
        document.writeln('<param name="iexplore_market_41" value="idc_dla|XCSE#KFMXPI|KFMX|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_41_types" value="1"/>');
		
        document.writeln('<param name="iexplore_market_42_path" value="Shares|Denmark - (delayed) "/>');
        document.writeln('<param name="iexplore_market_42" value="idc_dla|XFND|First North|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_42_types" value="1"/>');
 
        document.writeln('<param name="iexplore_market_43_path" value="Shares|Denmark - (delayed) "/>');
        document.writeln('<param name="iexplore_market_43" value="idc_dla|XCSE|All|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_43_types" value="1"/>');
		document.writeln('<param name="iexplore_market_43_group_by_letter" value="true;20" />');
//		document.writeln('<param name="iexplore_market_43_regex" value="0|^[^0-9]*$" />');

//Shares Finland Real-time

        document.writeln('<param name="iexplore_market_44_path" value="Shares|Finland - (real-time)"/>');
        document.writeln('<param name="iexplore_market_44"value="idc_rtb|CHIXH#OMXH25|OMXH25|false|true|true|false"/>');
        document.writeln('<param name="iexplore_market_44_types" value="1"/>');

        document.writeln('<param name="iexplore_market_45_path" value="Shares|Finland - (real-time)"/>');
        document.writeln('<param name="iexplore_market_45" value="idc_rtb|CHIXH|All|false|true|true|false"/>');
        document.writeln('<param name="iexplore_market_45_types" value="1"/>');
		document.writeln('<param name="iexplore_market_45_group_by_letter" value="true;20" />');
//		document.writeln('<param name="iexplore_market_45_regex" value="0|^[^0-9]*$" />');		
 
//Shares Finland Delayed

        document.writeln('<param name="iexplore_market_46_path" value="Shares|Finland - (delayed) "/>');
        document.writeln('<param name="iexplore_market_46"value="idc_dla|XHEL#OMXH25|OMXH25|false|true|true|false"/>');
        document.writeln('<param name="iexplore_market_46_types" value="1"/>');

        document.writeln('<param name="iexplore_market_47_path" value="Shares|Finland - (delayed) "/>');
        document.writeln('<param name="iexplore_market_47" value="idc_dla|FNFI|First North|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_47_types" value="1"/>');

        document.writeln('<param name="iexplore_market_48_path" value="Shares|Finland - (delayed) "/>');
        document.writeln('<param name="iexplore_market_48" value="idc_dla|XHEL|All|false|true|true|false"/>');
        document.writeln('<param name="iexplore_market_48_types" value="1"/>');
		document.writeln('<param name="iexplore_market_48_group_by_letter" value="true;20" />');
//		document.writeln('<param name="iexplore_market_48_regex" value="0|^[^0-9]*$" />');		
 
//Shares France Real-time

        document.writeln('<param name="iexplore_market_49_path" value="Shares|France - (real-time)" />');
        document.writeln('<param name="iexplore_market_49" value="idc_rtb|CHIXP#PX1|CAC 40|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_49_types" value="1" />');

        document.writeln('<param name="iexplore_market_50_path" value="Shares|France - (real-time)"/>');
        document.writeln('<param name="iexplore_market_50" value="idc_rtb|CHIXP#CACMD.en|CAC Mid|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_50_types" value="1" />');

        document.writeln('<param name="iexplore_market_51_path" value="Shares|France - (real-time)" />');
        document.writeln('<param name="iexplore_market_51" value="idc_rtb|CHIXP#CACS.en|CAC Small|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_51_types" value="1" />');

        document.writeln('<param name="iexplore_market_52_path" value="Shares|France - (real-time)" />');
        document.writeln('<param name="iexplore_market_52" value="idc_rtb|CHIXP#MS190|CAC  Mid & Small|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_52_types" value="1" />');
        
        document.writeln('<param name="iexplore_market_53_path" value="Shares|France - (real-time)" />');
        document.writeln('<param name="iexplore_market_53" value="idc_rtb|CHIXP#PXT|IT CAC|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_53_types" value="1" />');

        document.writeln('<param name="iexplore_market_54_path" value="Shares|France - (real-time)" />');
        document.writeln('<param name="iexplore_market_54" value="idc_rtb|CHIXP#PXC|CAC IT20|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_54_types" value="1" />');

        document.writeln('<param name="iexplore_market_55_path" value="Shares|France - (real-time)" />');
        document.writeln('<param name="iexplore_market_55" value="idc_rtb|CHIXP#CN20|CAC NEXT|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_55_types" value="1" />');

        document.writeln('<param name="iexplore_market_56_path" value="Shares|France - (real-time)" />');
        document.writeln('<param name="iexplore_market_56" value="idc_rtb|CHIXP#PX8|SBF80 index|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_56_types" value="1" />');

        document.writeln('<param name="iexplore_market_57_path" value="Shares|France - (real-time)" />');
        document.writeln('<param name="iexplore_market_57" value="idc_rtb|CHIXP#PX4|SBF120 index|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_57_types" value="1" />');
	
        document.writeln('<param name="iexplore_market_58_path" value="Shares|France - (real-time)" />');
        document.writeln('<param name="iexplore_market_58" value="idc_rtb|CHIXP#PX5|SBF250 index|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_58_types" value="1" />');

        document.writeln('<param name="iexplore_market_59_path" value="Shares|France - (real-time)" />');
        document.writeln('<param name="iexplore_market_59" value="idc_rtb|CHIXP#IAS|EURONEXT FAS IAS|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_59_types" value="1"  />');
       
        document.writeln('<param name="iexplore_market_60_path" value="Shares|France - (real-time)" />');
        document.writeln('<param name="iexplore_market_60" value="idc_rtb|CHIXP#SIIC|IND.S.I.I.C.FRANCE|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_60_types" value="1" />');

        document.writeln('<param name="iexplore_market_61_path" value="Shares|France - (real-time)" />');
		document.writeln('<param name="iexplore_market_name_61" value="CHIXP;France">');
        document.writeln('<param name="iexplore_market_61" value="idc_rtb|CHIXP|All|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_61_types" value="1" />');
//      document.writeln('<param name="iexplore_market_61_regex" value="0|^[^0-9]*$" />');
        document.writeln('<param name="iexplore_market_61_group_by_letter" value="true;50" />');
		
//Shares France Delayed
		
		document.writeln('<param name="iexplore_market_62_path" value="Shares|France - (delayed) " />');
        document.writeln('<param name="iexplore_market_62" value="idc_dlb|XPAR#PX1|CAC 40|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_62_types" value="1" />');

        document.writeln('<param name="iexplore_market_63_path" value="Shares|France - (delayed) "/>');
        document.writeln('<param name="iexplore_market_63" value="idc_dlb|XPAR#CACMD.en|CAC Mid|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_63_types" value="1" />');

        document.writeln('<param name="iexplore_market_64_path" value="Shares|France - (delayed) " />');
        document.writeln('<param name="iexplore_market_64" value="idc_dlb|XPAR#CACS.en|CAC Small|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_64_types" value="1" />');

        document.writeln('<param name="iexplore_market_65_path" value="Shares|France - (delayed) " />');
        document.writeln('<param name="iexplore_market_65" value="idc_dlb|XPAR#MS190|CAC  Mid & Small|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_65_types" value="1" />');
       
        document.writeln('<param name="iexplore_market_66_path" value="Shares|France - (delayed) " />');
        document.writeln('<param name="iexplore_market_66" value="idc_dlb|XPAR#PXT|IT CAC|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_66_types" value="1" />');

        document.writeln('<param name="iexplore_market_67_path" value="Shares|France - (delayed) " />');
        document.writeln('<param name="iexplore_market_67" value="idc_dlb|XPAR#PXC|CAC IT20|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_67_types" value="1" />');

        document.writeln('<param name="iexplore_market_68_path" value="Shares|France - (delayed) " />');
        document.writeln('<param name="iexplore_market_68" value="idc_dlb|XPAR#CN20|CAC NEXT|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_68_types" value="1" />');

        document.writeln('<param name="iexplore_market_69_path" value="Shares|France - (delayed) " />');
        document.writeln('<param name="iexplore_market_69" value="idc_dlb|XPAR#PX8|SBF80 index|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_69_types" value="1" />');

        document.writeln('<param name="iexplore_market_70_path" value="Shares|France - (delayed) " />');
        document.writeln('<param name="iexplore_market_70" value="idc_dlb|XPAR#PX4|SBF120 index|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_70_types" value="1" />');
		
        document.writeln('<param name="iexplore_market_71_path" value="Shares|France - (delayed) " />');
        document.writeln('<param name="iexplore_market_71" value="idc_dlb|XPAR#PX5|SBF250 index|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_71_types" value="1" />');

        document.writeln('<param name="iexplore_market_72_path" value="Shares|France - (delayed) " />');
        document.writeln('<param name="iexplore_market_72" value="idc_dlb|XPAR#IAS|EURONEXT FAS IAS|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_72_types" value="1"  />');
        
        document.writeln('<param name="iexplore_market_73_path" value="Shares|France - (delayed) " />');
        document.writeln('<param name="iexplore_market_73" value="idc_dlb|XPAR#SIIC|IND.S.I.I.C.FRANCE|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_73_types" value="1" />');

        document.writeln('<param name="iexplore_market_74_path" value="Shares|France - (delayed) " />');
   document.writeln('<param name="iexplore_market_name_74" value="XPAR;France">');
        document.writeln('<param name="iexplore_market_74" value="idc_dlb|XPAR|All|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_74_types" value="1" />');
//      document.writeln('<param name="iexplore_market_74_regex" value="0|^[^0-9]*$" />');
        document.writeln('<param name="iexplore_market_74_group_by_letter" value="true;50" />');

//Shares Germany Real-time

        document.writeln('<param name="iexplore_market_75_path" value="Shares|Germany - (real-time)"  />');
        document.writeln('<param name="iexplore_market_75" value="idc_rtb|CHIXD#DAX.de|DAX|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_75_types" value="1"  />');

		document.writeln('<param name="iexplore_market_76_path" value="Shares|Germany - (real-time)"  />');
   document.writeln('<param name="iexplore_market_name_76" value="CHIXD;Germany">');
        document.writeln('<param name="iexplore_market_76" value="idc_rtb|CHIXD|All|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_76_types" value="1"  />');
//       document.writeln('<param name="iexplore_market_76_regex" value="0|^[^0-9]*$"  />');
        document.writeln('<param name="iexplore_market_76_group_by_letter" value="true;20"  />');

//Shares India Delayed

        document.writeln('<param name="iexplore_market_77_path" value="Shares|India - (delayed)"  />');
        document.writeln('<param name="iexplore_market_77" value="idc_dlc|XNSE#NIFTY.in|Nifty 50|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_77_types" value="1"  />');

		document.writeln('<param name="iexplore_market_78_path" value="Shares|India - (delayed)"  />');
   document.writeln('<param name="iexplore_market_name_78" value="XNSE;India">');
        document.writeln('<param name="iexplore_market_78" value="idc_dlc|XNSE|All|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_78_types" value="1"  />');
//      document.writeln('<param name="iexplore_market_78_regex" value="0|^[^0-9]*$"  />');
        document.writeln('<param name="iexplore_market_78_group_by_letter" value="true;20"  />');
				

//Shares Ireland Real-time

        document.writeln('<param name="iexplore_market_79_path" value="Shares|Ireland - (real-time)"  />');
        document.writeln('<param name="iexplore_market_79" value="idc_rtb|CHIXI#ISEQ20|ISEQ20|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_79_types" value="1"  />');
	
		document.writeln('<param name="iexplore_market_80_path" value="Shares|Ireland - (real-time)"  />');
   document.writeln('<param name="iexplore_market_name_80" value="CHIXI;Ireland">');
        document.writeln('<param name="iexplore_market_80" value="idc_rtb|CHIXI|All|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_80_types" value="1"  />');
//      document.writeln('<param name="iexplore_market_80_regex" value="0|^[^0-9]*$"  />');
        document.writeln('<param name="iexplore_market_80_group_by_letter" value="true;20"  />');
		
//Shares Italy Real-time

        document.writeln('<param name="iexplore_market_81_path" value="Shares|Italy - (real-time)"  />');
        document.writeln('<param name="iexplore_market_81" value="idc_rtb|CHIXM#FTSEMIB|FTSE MIB|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_81_types" value="1"  />');
		
		document.writeln('<param name="iexplore_market_82_path" value="Shares|Italy - (real-time)"  />');
		document.writeln('<param name="iexplore_market_name_82" value="CHIXM;Italy">');
        document.writeln('<param name="iexplore_market_82" value="idc_rtb|CHIXM|All|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_82_types" value="1"  />');
 //     document.writeln('<param name="iexplore_market_82_regex" value="0|^[^0-9]*$"  />');
        document.writeln('<param name="iexplore_market_82_group_by_letter" value="true;20"  />');

//Shares Netherlands Real-time
		
        document.writeln('<param name="iexplore_market_83_path" value="Shares|Netherlands - (real-time)"  />');
        document.writeln('<param name="iexplore_market_83" value="idc_rtb|CHIXA#AEX.en|AEX|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_83_types" value="1"  />');

        document.writeln('<param name="iexplore_market_84_path" value="Shares|Netherlands - (real-time)"  />');
        document.writeln('<param name="iexplore_market_84" value="idc_rtb|CHIXA#AMX.en|AMX|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_84_types" value="1"  />');

        document.writeln('<param name="iexplore_market_85_path" value="Shares|Netherlands - (real-time)"  />');
        document.writeln('<param name="iexplore_market_85" value="idc_rtb|CHIXA#ASCX.en|AMS SMALL CAP INDX|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_85_types" value="1"  />');

        document.writeln('<param name="iexplore_market_86_path" value="Shares|Netherlands - (real-time)"  />');
   document.writeln('<param name="iexplore_market_name_86" value="CHIXA;Netherlands">');
        document.writeln('<param name="iexplore_market_86" value="idc_rtb|CHIXA|All|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_86_types" value="1"  />');
        document.writeln('<param name="iexplore_market_86_regex" value="0|^[^0-9]*$"  />');
        document.writeln('<param name="iexplore_market_86_group_by_letter" value="true;20"  />');

//Shares Netherlands Delayed

		document.writeln('<param name="iexplore_market_87_path" value="Shares|Netherlands - (delayed) "  />');
        document.writeln('<param name="iexplore_market_87" value="idc_dlb|XAMS#AEX.en|AEX|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_87_types" value="1"  />');

        document.writeln('<param name="iexplore_market_88_path" value="Shares|Netherlands - (delayed) "  />');
        document.writeln('<param name="iexplore_market_88" value="idc_dlb|XAMS#AMX.en|AMX|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_88_types" value="1"  />');

        document.writeln('<param name="iexplore_market_89_path" value="Shares|Netherlands - (delayed) "  />');
        document.writeln('<param name="iexplore_market_89" value="idc_dlb|XAMS#ASCX.en|AMS SMALL CAP INDX|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_89_types" value="1"  />');

        document.writeln('<param name="iexplore_market_90_path" value="Shares|Netherlands - (delayed) "  />');
   document.writeln('<param name="iexplore_market_name_90" value="XAMS;Netherlands">');
        document.writeln('<param name="iexplore_market_90" value="idc_dlb|XAMS|All|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_90_types" value="1"  />');
 //     document.writeln('<param name="iexplore_market_90_regex" value="0|^[^0-9]*$"  />');
        document.writeln('<param name="iexplore_market_90_group_by_letter" value="true;20"  />');


//Shares Norway Real-time

        document.writeln('<param name="iexplore_market_91_path" value="Shares|Norway - (real-time)"  />');
        document.writeln('<param name="iexplore_market_91" value="idc_rtb|CHIXO#OMXO20.st|OMXO20|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_91_types" value="1"  />');
		
		document.writeln('<param name="iexplore_market_92_path" value="Shares|Norway - (real-time)"  />');
        document.writeln('<param name="iexplore_market_92" value="idc_rtb|CHIXO#OBX|OBX|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_92_types" value="1"  />');

		document.writeln('<param name="iexplore_market_93_path" value="Shares|Norway - (real-time)"  />');
   document.writeln('<param name="iexplore_market_name_93" value="CHIXO;Norway">');
        document.writeln('<param name="iexplore_market_93" value="idc_rtb|CHIXO|All|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_93_types" value="1"  />');
//      document.writeln('<param name="iexplore_market_93_regex" value="0|^[^0-9]*$"  />');
        document.writeln('<param name="iexplore_market_93_group_by_letter" value="true;20"  />');

//Shares Portugal Real-time

        document.writeln('<param name="iexplore_market_94_path" value="Shares|Portugal - (real-time)"  />');
        document.writeln('<param name="iexplore_market_94" value="idc_rtb|CHIXU#PSI20.en|PSI 20|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_94_types" value="1"  />');
	
		document.writeln('<param name="iexplore_market_95_path" value="Shares|Portugal - (real-time)" />');
        document.writeln('<param name="iexplore_market_95" value="idc_rtb|CHIXU#BVLGR.en|PSI General|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_95_types" value="1" />');

		document.writeln('<param name="iexplore_market_96_path" value="Shares|Portugal - (real-time)"  />');
   document.writeln('<param name="iexplore_market_name_96" value="CHIXU;Portugal">');
        document.writeln('<param name="iexplore_market_96" value="idc_rtb|CHIXU|All|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_96_types" value="1"  />');
//      document.writeln('<param name="iexplore_market_96_regex" value="0|^[^0-9]*$"  />');
        document.writeln('<param name="iexplore_market_96_group_by_letter" value="true;20"  />');

//Shares Portugal Delayed
		
		document.writeln('<param name="iexplore_market_97_path" value="Shares|Portugal - (delayed) " />');
        document.writeln('<param name="iexplore_market_97" value="idc_dlb|XLIS#PSI20.en|PSI 20|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_97_types" value="1" />');
	
        document.writeln('<param name="iexplore_market_98_path" value="Shares|Portugal - (delayed) " />');
        document.writeln('<param name="iexplore_market_98" value="idc_dlb|XLIS#BVLGR.en|PSI General|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_98_types" value="1" />');

        document.writeln('<param name="iexplore_market_99_path" value="Shares|Portugal - (delayed) " />');
   document.writeln('<param name="iexplore_market_name_99" value="XLIS;Portugal">');
        document.writeln('<param name="iexplore_market_99" value="idc_dlb|XLIS|All|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_99_types" value="1" />');
//      document.writeln('<param name="iexplore_market_99_regex" value="0|^[^0-9]*$" />');
        document.writeln('<param name="iexplore_market_99_group_by_letter" value="true;20" />');

//Shares Spain Real-time

        document.writeln('<param name="iexplore_market_100_path" value="Shares|Spain - (real-time)"  />');
        document.writeln('<param name="iexplore_market_100" value="idc_rtb|CHIXE#IBEX|IBEX|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_100_types" value="1"  />');

		document.writeln('<param name="iexplore_market_101_path" value="Shares|Spain - (real-time)"  />');
   document.writeln('<param name="iexplore_market_name_101" value="CHIXE;Spain">');
        document.writeln('<param name="iexplore_market_101" value="idc_rtb|CHIXE|All|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_101_types" value="1"  />');
//      document.writeln('<param name="iexplore_market_101_regex" value="0|^[^0-9]*$"  />');
        document.writeln('<param name="iexplore_market_101_group_by_letter" value="true;20"  />');

//Shares Sweden Real-time
		
		document.writeln('<param name="iexplore_market_102_path" value="Shares|Sweden - (real-time)"/>');
        document.writeln('<param name="iexplore_market_102" value="idc_rtb|CHIXS#OMXS30|OMXS30|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_102_types" value="1"/>');


        document.writeln('<param name="iexplore_market_103_path" value="Shares|Sweden - (real-time)"/>');
        document.writeln('<param name="iexplore_market_103" value="idc_rtb|CHIXS|All|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_103_types" value="1"/>');
		document.writeln('<param name="iexplore_market_103_group_by_letter" value="true;20" />');
//		document.writeln('<param name="iexplore_market_103_regex" value="0|^[^0-9]*$" />');
	

//Shares Sweden Delayed
	
		document.writeln('<param name="iexplore_market_104_path" value="Shares|Sweden - (delayed) "/>');
        document.writeln('<param name="iexplore_market_104" value="idc_dla|XSTO#OMXS30|OMXS30|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_104_types" value="1"/>');

        document.writeln('<param name="iexplore_market_105_path" value="Shares|Sweden - (delayed) " />');
        document.writeln('<param name="iexplore_market_105" value="idc_dla|FNSE|First North|false|true|false|false" />');
        document.writeln('<param name="iexplore_market_105_types" value="1" />');
	
        document.writeln('<param name="iexplore_market_106_path" value="Shares|Sweden - (delayed) "/>');
        document.writeln('<param name="iexplore_market_106" value="idc_dla|XSTO|All|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_106_types" value="1"/>');
		document.writeln('<param name="iexplore_market_106_group_by_letter" value="true;20" />');
//		document.writeln('<param name="iexplore_market_106_regex" value="0|^[^0-9]*$" />');


//Shares Swiss  Real-time
		
		document.writeln('<param name="iexplore_market_107_path" value="Shares|Swiss - (real-time)"/>');
        document.writeln('<param name="iexplore_market_107" value="idc_rtb|CHIXZ#SMI.sw|SMI|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_107_types" value="1"/>');

		document.writeln('<param name="iexplore_market_108_path" value="Shares|Swiss - (real-time)"/>');
        document.writeln('<param name="iexplore_market_108" value="idc_rtb|CHIXZ#SMIM.sw|SMI Mid|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_108_types" value="1"/>');
		
		document.writeln('<param name="iexplore_market_109_path" value="Shares|Swiss - (real-time)"/>');
        document.writeln('<param name="iexplore_market_109" value="idc_rtb|CHIXZ#SMIEXP.sw|SMI Expanded|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_109_types" value="1"/>');
		
        document.writeln('<param name="iexplore_market_110_path" value="Shares|Swiss - (real-time)"/>');
        document.writeln('<param name="iexplore_market_110" value="idc_rtb|CHIXZ|All|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_110_types" value="1"/>');
		document.writeln('<param name="iexplore_market_110_group_by_letter" value="true;20" />');
//		document.writeln('<param name="iexplore_market_110_regex" value="0|^[^0-9]*$" />');

//Shares Turkey Delayed
	
		document.writeln('<param name="iexplore_market_111_path" value="Shares|Turkey - (delayed) "/>');
        document.writeln('<param name="iexplore_market_111" value="idc_dla|XIST#XU100|ISE National 100|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_111_types" value="1"/>');

        document.writeln('<param name="iexplore_market_112_path" value="Shares|Turkey - (delayed) "/>');
        document.writeln('<param name="iexplore_market_112" value="idc_dla|XIST|All|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_112_types" value="1"/>');
		document.writeln('<param name="iexplore_market_112_group_by_letter" value="true;20" />');
//		document.writeln('<param name="iexplore_market_112_regex" value="0|^[^0-9]*$" />');	

//Shares United Kingdom  Real-time
		
		document.writeln('<param name="iexplore_market_113_path" value="Shares|UK - (real-time)"/>');
        document.writeln('<param name="iexplore_market_113" value="idc_rtb|CHIXL#UKX|FTSE 100|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_113_types" value="1"/>');

        document.writeln('<param name="iexplore_market_114_path" value="Shares|UK - (real-time)"/>');
        document.writeln('<param name="iexplore_market_114" value="idc_rtb|CHIXL|All|false|true|false|false"/>');
        document.writeln('<param name="iexplore_market_114_types" value="1"/>');
		document.writeln('<param name="iexplore_market_114_group_by_letter" value="true;20" />');
//		document.writeln('<param name="iexplore_market_114_regex" value="0|^[^0-9]*$" />');
		

//Shares USA  Delayed

		document.writeln('<param name="iexplore_market_115_path" value="Shares|USA - (delayed) " />');
		document.writeln('<param name="iexplore_market_115" value="idc_dlb|XNYS#!DJI|Dow Jones Ind. constituents|false|true|false|false" />');
		document.writeln('<param name="iexplore_market_115_types" value="1" />');

		document.writeln('<param name="iexplore_market_116_path" value="Shares|USA - (delayed) " />');
		document.writeln('<param name="iexplore_market_116" value="idc_dlb|XNAS#NDX|Nasdaq 100 Constituents|false|true|false|false" />');
		document.writeln('<param name="iexplore_market_116_types" value="1" />');
				       
        document.writeln('<param name="iexplore_market_117_path" value="Shares|USA - (delayed) "  />');
 //document.writeln('<param name="iexplore_market_name_117" value="XNAS;All">');
        document.writeln('<param name="iexplore_market_117" value="idc_dlb|XNAS|Nasdaq|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_117_types" value="1"  />');
        document.writeln('<param name="iexplore_market_117_group_by_letter" value="true;50"  />');	

        document.writeln('<param name="iexplore_market_118_path" value="Shares|USA - (delayed) "  />');
 //document.writeln('<param name="iexplore_market_name_118" value="XNYS;All">');
        document.writeln('<param name="iexplore_market_118" value="idc_dlb|XNYS|NYSE|false|true|false|false"  />');
        document.writeln('<param name="iexplore_market_118_types" value="1"  />');
        document.writeln('<param name="iexplore_market_118_group_by_letter" value="true;50"  />');	
				
document.writeln('<param name="iexplore_market_119" value="@HTML|interbankfx_|Interbank FX - NetDania|false|true|false|false" />');
document.writeln('<param name="interbankfx_1" value="AUDCAD|netdania_rt;AUD/CAD;true;0"/>');
document.writeln('<param name="interbankfx_2" value="AUDCHF|netdania_rt;AUD/CHF;true;0"/>');
document.writeln('<param name="interbankfx_3" value="AUDJPY|netdania_rt;AUD/JPY;true;0"/>');
document.writeln('<param name="interbankfx_4" value="AUDNOK|netdania_rt;AUD/NOK;true;0"/>');
document.writeln('<param name="interbankfx_5" value="AUDNZD|netdania_rt;AUD/NZD;true;0"/>');
document.writeln('<param name="interbankfx_6" value="AUDSEK|netdania_rt;AUD/SEK;true;0"/>');
document.writeln('<param name="interbankfx_7" value="AUDUSD|netdania_rt;AUD/USD;true;0"/>');
document.writeln('<param name="interbankfx_8" value="BWPUSD|netdania_rt;BWP/USD;true;0"/>');
document.writeln('<param name="interbankfx_9" value="CADCHF|netdania_rt;CAD/CHF;true;0"/>');
document.writeln('<param name="interbankfx_10" value="CADJPY|netdania_rt;CAD/JPY;true;0"/>');
document.writeln('<param name="interbankfx_11" value="CHFJPY|netdania_rt;CHF/JPY;true;0"/>');
document.writeln('<param name="interbankfx_12" value="CHFNOK|netdania_rt;CHF/NOK;true;0"/>');
document.writeln('<param name="interbankfx_13" value="CHFPLN|netdania_rt;CHF/PLN;true;0"/>');
document.writeln('<param name="interbankfx_14" value="EURAUD|netdania_rt;EUR/AUD;true;0"/>');
document.writeln('<param name="interbankfx_15" value="EURCAD|netdania_rt;EUR/CAD;true;0"/>');
document.writeln('<param name="interbankfx_16" value="EURCHF|netdania_rt;EUR/CHF;true;0"/>');
document.writeln('<param name="interbankfx_17" value="EURCZK|netdania_rt;EUR/CZK;true;0"/>');
document.writeln('<param name="interbankfx_18" value="EURGBP|netdania_rt;EUR/GBP;true;0"/>');
document.writeln('<param name="interbankfx_19" value="EURHUF|netdania_rt;EUR/HUF;true;0"/>');
document.writeln('<param name="interbankfx_20" value="EURILS|netdania_rt;EUR/ILS;true;0"/>');
document.writeln('<param name="interbankfx_21" value="EURJPY|netdania_rt;EUR/JPY;true;0"/>');
document.writeln('<param name="interbankfx_22" value="EURMXN|netdania_rt;EUR/MXN;true;0"/>');
document.writeln('<param name="interbankfx_23" value="EURNOK|netdania_rt;EUR/NOK;true;0"/>');
document.writeln('<param name="interbankfx_24" value="EURNZD|netdania_rt;EUR/NZD;true;0"/>');
document.writeln('<param name="interbankfx_25" value="EURPLN|netdania_rt;EUR/PLN;true;0"/>');
document.writeln('<param name="interbankfx_26" value="EURSEK|netdania_rt;EUR/SEK;true;0"/>');
document.writeln('<param name="interbankfx_27" value="EURTRY|netdania_rt;EUR/TRY;true;0"/>');
document.writeln('<param name="interbankfx_28" value="EURUSD|netdania_rt;EUR/USD;true;0"/>');
document.writeln('<param name="interbankfx_29" value="EURZAR|netdania_rt;EUR/ZAR;true;0"/>');
document.writeln('<param name="interbankfx_30" value="GBPAUD|netdania_rt;GBP/AUD;true;0"/>');
document.writeln('<param name="interbankfx_31" value="GBPCAD|netdania_rt;GBP/CAD;true;0"/>');
document.writeln('<param name="interbankfx_32" value="GBPCHF|netdania_rt;GBP/CHF;true;0"/>');
document.writeln('<param name="interbankfx_33" value="GBPJPY|netdania_rt;GBP/JPY;true;0"/>');
document.writeln('<param name="interbankfx_34" value="GBPNOK|netdania_rt;GBP/NOK;true;0"/>');
document.writeln('<param name="interbankfx_35" value="GBPNZD|netdania_rt;GBP/NZD;true;0"/>');
document.writeln('<param name="interbankfx_36" value="GBPPLN|netdania_rt;GBP/PLN;true;0"/>');
document.writeln('<param name="interbankfx_37" value="GBPSEK|netdania_rt;GBP/SEK;true;0"/>');
document.writeln('<param name="interbankfx_38" value="GBPTRY|netdania_rt;GBP/TRY;true;0"/>');
document.writeln('<param name="interbankfx_39" value="GBPUSD|netdania_rt;GBP/USD;true;0"/>');
document.writeln('<param name="interbankfx_40" value="GBPZAR|netdania_rt;GBP/ZAR;true;0"/>');
document.writeln('<param name="interbankfx_41" value="NOKJPY|netdania_rt;NOK/JPY;true;0"/>');
document.writeln('<param name="interbankfx_42" value="NOKSEK|netdania_rt;NOK/SEK;true;0"/>');
document.writeln('<param name="interbankfx_43" value="NZDCAD|netdania_rt;NZD/CAD;true;0"/>');
document.writeln('<param name="interbankfx_44" value="NZDCHF|netdania_rt;NZD/CHF;true;0"/>');
document.writeln('<param name="interbankfx_45" value="NZDJPY|netdania_rt;NZD/JPY;true;0"/>');
document.writeln('<param name="interbankfx_46" value="NZDUSD|netdania_rt;NZD/USD;true;0"/>');
document.writeln('<param name="interbankfx_47" value="USDAED|netdania_rt;USD/AED;true;0"/>');
document.writeln('<param name="interbankfx_48" value="USDARS|netdania_rt;USD/ARS;true;0"/>');
document.writeln('<param name="interbankfx_49" value="USDBBD|netdania_rt;USD/BBD;true;0"/>');
document.writeln('<param name="interbankfx_50" value="USDBGN|netdania_rt;USD/BGN;true;0"/>');
document.writeln('<param name="interbankfx_51" value="USDBHD|netdania_rt;USD/BHD;true;0"/>');
document.writeln('<param name="interbankfx_52" value="USDBRL|netdania_rt;USD/BRL;true;0"/>');
document.writeln('<param name="interbankfx_53" value="USDCAD|netdania_rt;USD/CAD;true;0"/>');
document.writeln('<param name="interbankfx_54" value="USDCHF|netdania_rt;USD/CHF;true;0"/>');
document.writeln('<param name="interbankfx_55" value="USDCLP|netdania_rt;USD/CLP;true;0"/>');
document.writeln('<param name="interbankfx_56" value="USDCOP|netdania_rt;USD/COP;true;0"/>');
document.writeln('<param name="interbankfx_57" value="USDDKK|netdania_rt;USD/DKK;true;0"/>');
document.writeln('<param name="interbankfx_58" value="USDHKD|netdania_rt;USD/HKD;true;0"/>');
document.writeln('<param name="interbankfx_59" value="USDHRK|netdania_rt;USD/HRK;true;0"/>');
document.writeln('<param name="interbankfx_60" value="USDHUF|netdania_rt;USD/HUF;true;0"/>');
document.writeln('<param name="interbankfx_61" value="USDIDR|netdania_rt;USD/IDR;true;0"/>');
document.writeln('<param name="interbankfx_62" value="USDILS|netdania_rt;USD/ILS;true;0"/>');
document.writeln('<param name="interbankfx_63" value="USDJOD|netdania_rt;USD/JOD;true;0"/>');
document.writeln('<param name="interbankfx_64" value="USDJPY|netdania_rt;USD/JPY;true;0"/>');
document.writeln('<param name="interbankfx_65" value="USDKRW|netdania_rt;USD/KRW;true;0"/>');
document.writeln('<param name="interbankfx_66" value="USDKWD|netdania_rt;USD/KWD;true;0"/>');
document.writeln('<param name="interbankfx_67" value="USDLTL|netdania_rt;USD/LTL;true;0"/>');
document.writeln('<param name="interbankfx_68" value="USDMAD|netdania_rt;USD/MAD;true;0"/>');
document.writeln('<param name="interbankfx_69" value="USDMXN|netdania_rt;USD/MXN;true;0"/>');
document.writeln('<param name="interbankfx_70" value="USDMYR|netdania_rt;USD/MYR;true;0"/>');
document.writeln('<param name="interbankfx_71" value="USDNOK|netdania_rt;USD/NOK;true;0"/>');
document.writeln('<param name="interbankfx_72" value="USDOMR|netdania_rt;USD/OMR;true;0"/>');
document.writeln('<param name="interbankfx_73" value="USDPHP|netdania_rt;USD/PHP;true;0"/>');
document.writeln('<param name="interbankfx_74" value="USDPKR|netdania_rt;USD/PKR;true;0"/>');
document.writeln('<param name="interbankfx_75" value="USDPLN|netdania_rt;USD/PLN;true;0"/>');
document.writeln('<param name="interbankfx_76" value="USDQAR|netdania_rt;USD/QAR;true;0"/>');
document.writeln('<param name="interbankfx_77" value="USDRUB|netdania_rt;USD/RUB;true;0"/>');
document.writeln('<param name="interbankfx_78" value="USDSAR|netdania_rt;USD/SAR;true;0"/>');
document.writeln('<param name="interbankfx_79" value="USDSEK|netdania_rt;USD/SEK;true;0"/>');
document.writeln('<param name="interbankfx_80" value="USDSEK|netdania_rt;USD/SEK;true;0"/>');
document.writeln('<param name="interbankfx_81" value="USDSGD|netdania_rt;USD/SGD;true;0"/>');
document.writeln('<param name="interbankfx_82" value="USDTHB|netdania_rt;USD/THB;true;0"/>');
document.writeln('<param name="interbankfx_83" value="USDTND|netdania_rt;USD/TND;true;0"/>');
document.writeln('<param name="interbankfx_84" value="USDTRY|netdania_rt;USD/TRY;true;0"/>');
document.writeln('<param name="interbankfx_85" value="USDTWD|netdania_rt;USD/TWD;true;0"/>');
document.writeln('<param name="interbankfx_86" value="USDZAR|netdania_rt;USD/ZAR;true;0"/>');


		document.writeln('<param name="iexplore_market_120" value="@HTML|tf_|Major Currencies - IDC|false|true|false|false" />');
		document.writeln('<param name="tf_1" value="AUDCAD|idc_lite;AUD/CAD;valuta;0" />');
		document.writeln('<param name="tf_2" value="AUDCHF|idc_lite;AUD/CHF;valuta;0" />');
		document.writeln('<param name="tf_3" value="AUDEUR|idc_lite;AUD/EUR;valuta;0" />');
		document.writeln('<param name="tf_4" value="AUDJPY|idc_lite;AUD/JPY;valuta;0" />');
		document.writeln('<param name="tf_5" value="AUDNZD|idc_lite;AUD/NZD;valuta;0" />');
		document.writeln('<param name="tf_6" value="AUDSGD|idc_lite;AUD/SGD;valuta;0" />');
		document.writeln('<param name="tf_7" value="AUDUSD|idc_lite;AUD/USD;valuta;0" />');
		document.writeln('<param name="tf_8" value="CADJPY|idc_lite;CAD/JPY;valuta;0" />');
		document.writeln('<param name="tf_9" value="CHFJPY|idc_lite;CHF/JPY;valuta;0" />');
		document.writeln('<param name="tf_10" value="EURAUD|idc_lite;EUR/AUD;valuta;0" />');
		document.writeln('<param name="tf_11" value="EURCHF|idc_lite;EUR/CHF;valuta;0" />');
		document.writeln('<param name="tf_12" value="EURGBP|idc_lite;EUR/GBP;valuta;0" />');
		document.writeln('<param name="tf_13" value="EURJPY|idc_lite;EUR/JPY;valuta;0" />');
		document.writeln('<param name="tf_14" value="EURUSD|idc_lite;EUR/USD;valuta;0" />');
		document.writeln('<param name="tf_15" value="GBPCHF|idc_lite;GBP/CHF;valuta;0" />');
		document.writeln('<param name="tf_16" value="GBPJPY|idc_lite;GBP/JPY;valuta;0" />');
		document.writeln('<param name="tf_17" value="GBPUSD|idc_lite;GBP/USD;valuta;0" />');
		document.writeln('<param name="tf_18" value="NZDJPY|idc_lite;NZD/JPY;valuta;0" />');
		document.writeln('<param name="tf_19" value="NZDUSD|idc_lite;NZD/USD;valuta;0" />');
		document.writeln('<param name="tf_20" value="USDCAD|idc_lite;USD/CAD;valuta;0" />');
		document.writeln('<param name="tf_21" value="USDCHF|idc_lite;USD/CHF;valuta;0" />');
		document.writeln('<param name="tf_22" value="USDHKD|idc_lite;USD/HKD;valuta;0" />');
		document.writeln('<param name="tf_23" value="USDJPY|idc_lite;USD/JPY;valuta;0" />');
		document.writeln('<param name="tf_24" value="USDSGD|idc_lite;USD/SGD;valuta;0" />');

		document.writeln('<param name="iexplore_market_121" value="@HTML|afn_|Afghanistan Afghani|false|true|false|false" />');
		document.writeln('<param name="afn_1" value="AFNUSD|idc_lite;AFN/USD;Valuta;0" />');	
		document.writeln('<param name="afn_2" value="CADAFN|idc_lite;CAD/AFN;Valuta;0" />');
		document.writeln('<param name="afn_3" value="CHFAFN|idc_lite;CHF/AFN;Valuta;0" />');
		document.writeln('<param name="afn_4" value="EURAFN|idc_lite;EUR/AFN;Valuta;0" />');
		document.writeln('<param name="afn_5" value="GBPAFN|idc_lite;GBP/AFN;Valuta;0" />');
		document.writeln('<param name="afn_6" value="JPYAFN|idc_lite;JPY/AFN;Valuta;0" />');
		document.writeln('<param name="afn_7" value="USDAFN|idc_lite;USD/AFN;Valuta;0" />');
		document.writeln('<param name="iexplore_market_121_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_122" value="@HTML|ars_|Argentine Peso|false|true|false|false" />');
		document.writeln('<param name="ars_1" value="ARSBRL|idc_lite;ARS/BRL;Valuta;0" />');	
		document.writeln('<param name="ars_2" value="ARSEUR|idc_lite;ARS/EUR;Valuta;0" />');	
		document.writeln('<param name="ars_3" value="ARSXCU|idc_lite;ARS/XCU;Valuta;0" />');	
		document.writeln('<param name="ars_4" value="AUDARS|idc_lite;AUD/ARS;Valuta;0" />');	
		document.writeln('<param name="ars_5" value="BRLARS|idc_lite;BRL/ARS;Valuta;0" />');	
		document.writeln('<param name="ars_6" value="CADARS|idc_lite;CAD/ARS;Valuta;0" />');
		document.writeln('<param name="ars_7" value="CHFARS|idc_lite;CHF/ARS;Valuta;0" />');	
		document.writeln('<param name="ars_8" value="EURARS|idc_lite;EUR/ARS;Valuta;0" />');	
		document.writeln('<param name="ars_9" value="GBPARS|idc_lite;GBP/ARS;Valuta;0" />');	
		document.writeln('<param name="ars_10" value="JPYARS|idc_lite;JPY/ARS;Valuta;0" />');	
		document.writeln('<param name="ars_11" value="SGDARS|idc_lite;SGD/ARS;Valuta;0" />');	
		document.writeln('<param name="ars_12" value="USDARS|idc_lite;USD/ARS;Valuta;0" />');			
		document.writeln('<param name="iexplore_market_122_path" value="Cross Rates" />');

		
		document.writeln('<param name="iexplore_market_123" value="@HTML|aud_|Australia Dollar|false|true|false|false" />');
		document.writeln('<param name="aud_1" value="AUDCAD|idc_lite;AUD/CAD;Valuta;0" />');
		document.writeln('<param name="aud_2" value="AUDCHF|idc_lite;AUD/CHF;Valuta;0" />');
		document.writeln('<param name="aud_3" value="AUDILS|idc_lite;AUD/ILS;Valuta;0" />');
		document.writeln('<param name="aud_4" value="AUDJPY|idc_lite;AUD/JPY;Valuta;0" />');
		document.writeln('<param name="aud_5" value="AUDNZD|idc_lite;AUD/NZD;Valuta;0" />');
		document.writeln('<param name="aud_6" value="AUDUSD|idc_lite;AUD/USD;Valuta;0" />');
		document.writeln('<param name="aud_7" value="EURAUD|idc_lite;EUR/AUD;Valuta;0" />');
		document.writeln('<param name="aud_8" value="GBPAUD|idc_lite;GBP/AUD;Valuta;0" />');
		document.writeln('<param name="aud_9" value="AUDAED|idc_lite;AUD/AED;Valuta;0" />');
		document.writeln('<param name="aud_10" value="AUDANG|idc_lite;AUD/ANG;Valuta;0" />');
		document.writeln('<param name="aud_11" value="AUDARS|idc_lite;AUD/ARS;Valuta;0" />');
		document.writeln('<param name="aud_12" value="AUDAWG|idc_lite;AUD/AWG;Valuta;0" />');
		document.writeln('<param name="aud_13" value="AUDBBD|idc_lite;AUD/BBD;Valuta;0" />');
		document.writeln('<param name="aud_14" value="AUDBHD|idc_lite;AUD/BHD;Valuta;0" />');
		document.writeln('<param name="aud_15" value="AUDBMD|idc_lite;AUD/BMD;Valuta;0" />');
		document.writeln('<param name="aud_16" value="AUDBOB|idc_lite;AUD/BOB;Valuta;0" />');
		document.writeln('<param name="aud_17" value="AUDBRL|idc_lite;AUD/BRL;Valuta;0" />');
		document.writeln('<param name="aud_18" value="AUDBSD|idc_lite;AUD/BSD;Valuta;0" />');
		document.writeln('<param name="aud_19" value="AUDBZD|idc_lite;AUD/BZD;Valuta;0" />');		
		document.writeln('<param name="aud_20" value="AUDCLP|idc_lite;AUD/CLP;Valuta;0" />');
		document.writeln('<param name="aud_21" value="AUDCNH|idc_lite;AUD/CNH;Valuta;0" />');
		document.writeln('<param name="aud_22" value="AUDCNY|idc_lite;AUD/CNY;Valuta;0" />');		
		document.writeln('<param name="aud_23" value="AUDCOP|idc_lite;AUD/COP;Valuta;0" />');
		document.writeln('<param name="aud_24" value="AUDCRC|idc_lite;AUD/CRC;Valuta;0" />');
		document.writeln('<param name="aud_25" value="AUDCZK|idc_lite;AUD/CZK;Valuta;0" />');		
		document.writeln('<param name="aud_26" value="AUDDKK|idc_lite;AUD/DKK;Valuta;0" />');
		document.writeln('<param name="aud_27" value="AUDDOP|idc_lite;AUD/DOP;Valuta;0" />');
		document.writeln('<param name="aud_28" value="AUDEGP|idc_lite;AUD/EGP;Valuta;0" />');	
		document.writeln('<param name="aud_29" value="AUDERN|idc_lite;AUD/ERN;Valuta;0" />');
		document.writeln('<param name="aud_30" value="AUDEUR|idc_lite;AUD/EUR;Valuta;0" />');		
		document.writeln('<param name="aud_31" value="AUDGBP|idc_lite;AUD/GBP;Valuta;0" />');
		document.writeln('<param name="aud_32" value="AUDGIP|idc_lite;AUD/GIP;Valuta;0" />');		
		document.writeln('<param name="aud_33" value="AUDGTQ|idc_lite;AUD/GTQ;Valuta;0" />');
		document.writeln('<param name="aud_34" value="AUDHKD|idc_lite;AUD/HKD;Valuta;0" />');		
		document.writeln('<param name="aud_35" value="AUDHNL|idc_lite;AUD/HNL;Valuta;0" />');
		document.writeln('<param name="aud_36" value="AUDHRK|idc_lite;AUD/HRK;Valuta;0" />');		
		document.writeln('<param name="aud_37" value="AUDHUF|idc_lite;AUD/HUF;Valuta;0" />');
		document.writeln('<param name="aud_38" value="AUDIDR|idc_lite;AUD/IDR;Valuta;0" />');		
		document.writeln('<param name="aud_39" value="AUDINR|idc_lite;AUD/INR;Valuta;0" />');
		document.writeln('<param name="aud_40" value="AUDJMD|idc_lite;AUD/JMD;Valuta;0" />');		
		document.writeln('<param name="aud_41" value="AUDJOD|idc_lite;AUD/JOD;Valuta;0" />');
		document.writeln('<param name="aud_42" value="AUDKES|idc_lite;AUD/KES;Valuta;0" />');			
		document.writeln('<param name="aud_43" value="AUDKRW|idc_lite;AUD/KRW;Valuta;0" />');
		document.writeln('<param name="aud_44" value="AUDKWD|idc_lite;AUD/KWD;Valuta;0" />');		
		document.writeln('<param name="aud_45" value="AUDKYD|idc_lite;AUD/KYD;Valuta;0" />');
		document.writeln('<param name="aud_46" value="AUDMAD|idc_lite;AUD/MAD;Valuta;0" />');		
		document.writeln('<param name="aud_47" value="AUDMGA|idc_lite;AUD/MGA;Valuta;0" />');
		document.writeln('<param name="aud_48" value="AUDMXN|idc_lite;AUD/MXN;Valuta;0" />');		
		document.writeln('<param name="aud_49" value="AUDMYR|idc_lite;AUD/MYR;Valuta;0" />');
		document.writeln('<param name="aud_50" value="AUDNOK|idc_lite;AUD/NOK;Valuta;0" />');		
		document.writeln('<param name="aud_51" value="AUDOMR|idc_lite;AUD/OMR;Valuta;0" />');
		document.writeln('<param name="aud_52" value="AUDPEN|idc_lite;AUD/PEN;Valuta;0" />');		
		document.writeln('<param name="aud_53" value="AUDPHP|idc_lite;AUD/PHP;Valuta;0" />');
		document.writeln('<param name="aud_54" value="AUDPKR|idc_lite;AUD/PKR;Valuta;0" />');		
		document.writeln('<param name="aud_55" value="AUDPLN|idc_lite;AUD/PLN;Valuta;0" />');		
		document.writeln('<param name="aud_56" value="AUDPYG|idc_lite;AUD/PYG;Valuta;0" />');
		document.writeln('<param name="aud_57" value="AUDQAR|idc_lite;AUD/QAR;Valuta;0" />');		
		document.writeln('<param name="aud_58" value="AUDRON|idc_lite;AUD/RON;Valuta;0" />');
		document.writeln('<param name="aud_59" value="AUDRSD|idc_lite;AUD/RSD;Valuta;0" />');				
		document.writeln('<param name="aud_60" value="AUDRUB|idc_lite;AUD/RUB;Valuta;0" />');		
		document.writeln('<param name="aud_61" value="AUDSAR|idc_lite;AUD/SAR;Valuta;0" />');
		document.writeln('<param name="aud_62" value="AUDSCR|idc_lite;AUD/SCR;Valuta;0" />');		
		document.writeln('<param name="aud_63" value="AUDSEK|idc_lite;AUD/SEK;Valuta;0" />');		
		document.writeln('<param name="aud_64" value="AUDSGD|idc_lite;AUD/SGD;Valuta;0" />');
		document.writeln('<param name="aud_65" value="AUDSVC|idc_lite;AUD/SVC;Valuta;0" />');		
		document.writeln('<param name="aud_66" value="AUDSYP|idc_lite;AUD/SYP;Valuta;0" />');		
		document.writeln('<param name="aud_67" value="AUDTHB|idc_lite;AUD/THB;Valuta;0" />');
		document.writeln('<param name="aud_68" value="AUDTJS|idc_lite;AUD/TJS;Valuta;0" />');				
		document.writeln('<param name="aud_69" value="AUDTND|idc_lite;AUD/TND;Valuta;0" />');		
		document.writeln('<param name="aud_70" value="AUDTRY|idc_lite;AUD/TRY;Valuta;0" />');
		document.writeln('<param name="aud_71" value="AUDTTD|idc_lite;AUD/TTD;Valuta;0" />');	
		document.writeln('<param name="aud_72" value="AUDTWD|idc_lite;AUD/TWD;Valuta;0" />');		
		document.writeln('<param name="aud_73" value="AUDUYU|idc_lite;AUD/UYU;Valuta;0" />');
		document.writeln('<param name="aud_74" value="AUDVEF|idc_lite;AUD/VEF;Valuta;0" />');	
		document.writeln('<param name="aud_75" value="AUDXCD|idc_lite;AUD/XCD;Valuta;0" />');		
		document.writeln('<param name="aud_76" value="AUDXCU|idc_lite;AUD/XCU;Valuta;0" />');
		document.writeln('<param name="aud_77" value="AUDZAR|idc_lite;AUD/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_123_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_124" value="@HTML|bhd_|Bahrain Dinar|false|true|false|false" />');
		document.writeln('<param name="bhd_1" value="EURBHD|idc_lite;EUR/BHD;Valuta;0" />');
		document.writeln('<param name="bhd_2" value="GBPBHD|idc_lite;GBP/BHD;Valuta;0" />');
		document.writeln('<param name="bhd_3" value="USDBHD|idc_lite;USD/BHD;Valuta;0" />');
		document.writeln('<param name="bhd_4" value="BHDCHF|idc_lite;BHD/CHF;Valuta;0" />');	
		document.writeln('<param name="bhd_5" value="BHDEUR|idc_lite;BHD/EUR;Valuta;0" />');	
		document.writeln('<param name="bhd_6" value="BHDGBP|idc_lite;BHD/GBP;Valuta;0" />');	
		document.writeln('<param name="bhd_7" value="BHDHKD|idc_lite;BHD/HKD;Valuta;0" />');	
		document.writeln('<param name="bhd_8" value="BHDNZD|idc_lite;BHD/NZD;Valuta;0" />');	
		document.writeln('<param name="bhd_9" value="BHDPKR|idc_lite;BHD/PKR;Valuta;0" />');	
		document.writeln('<param name="bhd_10" value="BHDSGD|idc_lite;BHD/SGD;Valuta;0" />');	
		document.writeln('<param name="bhd_11" value="BHDUSD|idc_lite;BHD/USD;Valuta;0" />');			
		document.writeln('<param name="iexplore_market_124_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_125" value="@HTML|bbd_|Barbadian Dollar|false|true|false|false" />');
		document.writeln('<param name="bbd_1" value="BBDGBP|idc_lite;BBD/GBP;Valuta;0" />');
		document.writeln('<param name="bbd_2" value="AUDBBD|idc_lite;AUD/BBD;Valuta;0" />');
		document.writeln('<param name="bbd_3" value="CADBBD|idc_lite;CAD/BBD;Valuta;0" />');
		document.writeln('<param name="bbd_4" value="GBPBBD|idc_lite;GBP/BBD;Valuta;0" />');
		document.writeln('<param name="bbd_5" value="JPYBBD|idc_lite;JPY/BBD;Valuta;0" />');
		document.writeln('<param name="bbd_6" value="USDBBD|idc_lite;USD/BBD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_125_path" value="Cross Rates" />');
	
		document.writeln('<param name="iexplore_market_126" value="@HTML|bmd_|Bermudian Dollar|false|true|false|false" />');
		document.writeln('<param name="bmd_1" value="BMDEUR|idc_lite;BMD/EUR;Valuta;0" />');
		document.writeln('<param name="bmd_2" value="AUDBMD|idc_lite;AUD/BMD;Valuta;0" />');
		document.writeln('<param name="bmd_3" value="CADBMD|idc_lite;CAD/BMD;Valuta;0" />');
		document.writeln('<param name="bmd_4" value="GBPBMD|idc_lite;GBP/BMD;Valuta;0" />');
		document.writeln('<param name="bmd_5" value="JPYBMD|idc_lite;JPY/BMD;Valuta;0" />');
		document.writeln('<param name="bmd_6" value="USDBMD|idc_lite;USD/BMD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_126_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_127" value="@HTML|bob_|Bolivian Boliviano|false|true|false|false" />');
		document.writeln('<param name="bob_1" value="BOBBRL|idc_lite;BOB/BRL;Valuta;0" />');
		document.writeln('<param name="bob_2" value="AUDBOB|idc_lite;AUD/BOB;Valuta;0" />');
		document.writeln('<param name="bob_3" value="BRLBOB|idc_lite;BRL/BOB;Valuta;0" />');
		document.writeln('<param name="bob_4" value="CADBOB|idc_lite;CAD/BOB;Valuta;0" />');
		document.writeln('<param name="bob_5" value="CHFBOB|idc_lite;CHF/BOB;Valuta;0" />');
		document.writeln('<param name="bob_6" value="EURBOB|idc_lite;EUR/BOB;Valuta;0" />');
		document.writeln('<param name="bob_7" value="GBPBOB|idc_lite;GBP/BOB;Valuta;0" />');
		document.writeln('<param name="bob_8" value="JPYBOB|idc_lite;JPY/BOB;Valuta;0" />');
		document.writeln('<param name="bob_9" value="USDBOB|idc_lite;USD/BOB;Valuta;0" />');
		document.writeln('<param name="iexplore_market_127_path" value="Cross Rates" />');	
		
		document.writeln('<param name="iexplore_market_128" value="@HTML|bam_|Bosnia - Herzogovinia Mark|false|true|false|false" />');
		document.writeln('<param name="bam_1" value="EURBAM|idc_lite;EUR/BAM;Valuta;0" />');
		document.writeln('<param name="bam_2" value="CHFBAM|idc_lite;CHF/BAM;Valuta;0" />');
		document.writeln('<param name="bam_3" value="DKKBAM|idc_lite;DKK/BAM;Valuta;0" />');
		document.writeln('<param name="bam_4" value="EURBAM|idc_lite;EUR/BAM;Valuta;0" />');
		document.writeln('<param name="bam_5" value="GBPBAM|idc_lite;GBP/BAM;Valuta;0" />');
		document.writeln('<param name="bam_6" value="USDBAM|idc_lite;USD/BAM;Valuta;0" />');
		document.writeln('<param name="iexplore_market_128_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_129" value="@HTML|bwp_|Botswana Pula|false|true|false|false" />');
		document.writeln('<param name="bwp_1" value="BWPEUR|idc_lite;BWP/EUR;Valuta;0" />');
		document.writeln('<param name="bwp_2" value="BWPUSD|idc_lite;BWP/USD;Valuta;0" />');
		document.writeln('<param name="bwp_3" value="BWPZAR|idc_lite;BWP/ZAR;Valuta;0" />');
		document.writeln('<param name="bwp_4" value="CADBWP|idc_lite;CAD/BWP;Valuta;0" />');
		document.writeln('<param name="bwp_5" value="CHFBWP|idc_lite;CHF/BWP;Valuta;0" />');
		document.writeln('<param name="bwp_6" value="EURBWP|idc_lite;EUR/BWP;Valuta;0" />');
		document.writeln('<param name="bwp_7" value="GBPBWP|idc_lite;GBP/BWP;Valuta;0" />');
		document.writeln('<param name="bwp_8" value="JPYBWP|idc_lite;JPY/BWP;Valuta;0" />');
		document.writeln('<param name="bwp_9" value="USDBWP|idc_lite;USD/BWP;Valuta;0" />');
		document.writeln('<param name="iexplore_market_129_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_130" value="@HTML|brl_|Brazilian Real|false|true|false|false" />');
		document.writeln('<param name="brl_1" value="EURBRL|idc_lite;EUR/BRL;Valuta;0" />');
		document.writeln('<param name="brl_2" value="GBPBRL|idc_lite;GBP/BRL;Valuta;0" />');
		document.writeln('<param name="brl_3" value="USDBRL|idc_lite;USD/BRL;Valuta;0" />');
		document.writeln('<param name="brl_4" value="BRLARS|idc_lite;BRL/ARS;Valuta;0" />');
		document.writeln('<param name="brl_5" value="BRLAUD|idc_lite;BRL/AUD;Valuta;0" />');
		document.writeln('<param name="brl_6" value="BRLBOB|idc_lite;BRL/BOB;Valuta;0" />');
		document.writeln('<param name="brl_7" value="BRLCHF|idc_lite;BRL/CHF;Valuta;0" />');		
		document.writeln('<param name="brl_8" value="BRLCLP|idc_lite;BRL/CLP;Valuta;0" />');
		document.writeln('<param name="brl_9" value="BRLCNY|idc_lite;BRL/CNY;Valuta;0" />');
		document.writeln('<param name="brl_10" value="BRLCOP|idc_lite;BRL/COP;Valuta;0" />');
		document.writeln('<param name="brl_11" value="BRLEUR|idc_lite;BRL/EUR;Valuta;0" />');				
		document.writeln('<param name="brl_12" value="BRLGBP|idc_lite;BRL/GBP;Valuta;0" />');
		document.writeln('<param name="brl_13" value="BRLHKD|idc_lite;BRL/HKD;Valuta;0" />');
		document.writeln('<param name="brl_14" value="BRLIDR|idc_lite;BRL/IDR;Valuta;0" />');
		document.writeln('<param name="brl_15" value="BRLILS|idc_lite;BRL/ILS;Valuta;0" />');				
		document.writeln('<param name="brl_16" value="BRLJPY|idc_lite;BRL/JPY;Valuta;0" />');
		document.writeln('<param name="brl_17" value="BRLMXN|idc_lite;BRL/MXN;Valuta;0" />');
		document.writeln('<param name="brl_18" value="BRLNZD|idc_lite;BRL/NZD;Valuta;0" />');
		document.writeln('<param name="brl_19" value="BRLRUB|idc_lite;BRL/RUB;Valuta;0" />');				
		document.writeln('<param name="brl_20" value="BRLUSD|idc_lite;BRL/USD;Valuta;0" />');
		document.writeln('<param name="brl_21" value="BRLVEF|idc_lite;BRL/VEF;Valuta;0" />');
		document.writeln('<param name="brl_22" value="BRLXCU|idc_lite;BRL/XCU;Valuta;0" />');				
		document.writeln('<param name="brl_23" value="BRLZAR|idc_lite;BRL/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_130_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_131" value="@HTML|bgn_|Bulgarian Lev|false|true|false|false" />');
		document.writeln('<param name="bgn_1" value="EURBGN|idc_lite;EUR/BGN;Valuta;0" />');
		document.writeln('<param name="bgn_2" value="GBPBGN|idc_lite;GBP/BGN;Valuta;0" />');
		document.writeln('<param name="bgn_3" value="USDBGN|idc_lite;USD/BGN;Valuta;0" />');
		document.writeln('<param name="bgn_4" value="BGNCHF|idc_lite;BGN/CHF;Valuta;0" />');
		document.writeln('<param name="bgn_5" value="BGNEUR|idc_lite;BGN/EUR;Valuta;0" />');		
		document.writeln('<param name="iexplore_market_131_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_132" value="@HTML|mmk_|Burmese Kyat|false|true|false|false" />');
		document.writeln('<param name="mmk_1" value="MMKUSD|idc_lite;MMK/USD;Valuta;0" />');
		document.writeln('<param name="mmk_2" value="CADMMK|idc_lite;CAD/MMK;Valuta;0" />');	
		document.writeln('<param name="mmk_3" value="CHFMMK|idc_lite;CHF/MMK;Valuta;0" />');	
		document.writeln('<param name="mmk_4" value="EURMMK|idc_lite;EUR/MMK;Valuta;0" />');	
		document.writeln('<param name="mmk_5" value="GBPMMK|idc_lite;GBP/MMK;Valuta;0" />');	
		document.writeln('<param name="mmk_6" value="JPYMMK|idc_lite;JPY/MMK;Valuta;0" />');	
		document.writeln('<param name="mmk_7" value="USDMMK|idc_lite;USD/MMK;Valuta;0" />');			
		document.writeln('<param name="iexplore_market_132_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_133" value="@HTML|cad_|Canadian Dollar|false|true|false|false" />');
		document.writeln('<param name="cad_1" value="CADAED|idc_lite;CAD/AED;Valuta;0" />');
		document.writeln('<param name="cad_2" value="CADAFN|idc_lite;CAD/AFN;Valuta;0" />');
		document.writeln('<param name="cad_3" value="CADANG|idc_lite;CAD/ANG;Valuta;0" />');
		document.writeln('<param name="cad_4" value="CADARS|idc_lite;CAD/ARS;Valuta;0" />');		
		document.writeln('<param name="cad_5" value="CADAUD|idc_lite;CAD/AUD;Valuta;0" />');
		document.writeln('<param name="cad_6" value="CADCHF|idc_lite;CAD/CHF;Valuta;0" />');
		document.writeln('<param name="cad_7" value="CADILS|idc_lite;CAD/ILS;Valuta;0" />');
		document.writeln('<param name="cad_8" value="CADJPY|idc_lite;CAD/JPY;Valuta;0" />');
		document.writeln('<param name="cad_9" value="EURCAD|idc_lite;EUR/CAD;Valuta;0" />');
		document.writeln('<param name="cad_10" value="GBPCAD|idc_lite;GBP/CAD;Valuta;0" />');
		document.writeln('<param name="cad_11" value="USDCAD|idc_lite;USD/CAD;Valuta;0" />');
		document.writeln('<param name="cad_12" value="CADAZN|idc_lite;CAD/AZN;Valuta;0" />');
		document.writeln('<param name="cad_13" value="CADBBD|idc_lite;CAD/BBD;Valuta;0" />');
		document.writeln('<param name="cad_14" value="CADBDT|idc_lite;CAD/BDT;Valuta;0" />');
		document.writeln('<param name="cad_15" value="CADBGN|idc_lite;CAD/BGN;Valuta;0" />');
		document.writeln('<param name="cad_16" value="CADBHD|idc_lite;CAD/BHD;Valuta;0" />');
		document.writeln('<param name="cad_17" value="CADBIF|idc_lite;CAD/BIF;Valuta;0" />');
		document.writeln('<param name="cad_18" value="CADBMD|idc_lite;CAD/BMD;Valuta;0" />');
		document.writeln('<param name="cad_19" value="CADBND|idc_lite;CAD/BND;Valuta;0" />');
		document.writeln('<param name="cad_20" value="CADBOB|idc_lite;CAD/BOB;Valuta;0" />');
		document.writeln('<param name="cad_21" value="CADBRL|idc_lite;CAD/BRL;Valuta;0" />');		
		document.writeln('<param name="cad_22" value="CADBSD|idc_lite;CAD/BSD;Valuta;0" />');
		document.writeln('<param name="cad_23" value="CADBTN|idc_lite;CAD/BTN;Valuta;0" />');
		document.writeln('<param name="cad_24" value="CADBWP|idc_lite;CAD/BWP;Valuta;0" />');
		document.writeln('<param name="cad_25" value="CADBYR|idc_lite;CAD/BYR;Valuta;0" />');
		document.writeln('<param name="cad_26" value="CADBZD|idc_lite;CAD/BZD;Valuta;0" />');
		document.writeln('<param name="cad_27" value="CADCDF|idc_lite;CAD/CDF;Valuta;0" />');
		document.writeln('<param name="cad_28" value="CADCHF|idc_lite;CAD/CHF;Valuta;0" />');
		document.writeln('<param name="cad_29" value="CADCLP|idc_lite;CAD/CLP;Valuta;0" />');
		document.writeln('<param name="cad_30" value="CADCNH|idc_lite;CAD/CNH;Valuta;0" />');
		document.writeln('<param name="cad_31" value="CADCNY|idc_lite;CAD/CNY;Valuta;0" />');			
		document.writeln('<param name="cad_32" value="CADCOP|idc_lite;CAD/COP;Valuta;0" />');
		document.writeln('<param name="cad_33" value="CADCRC|idc_lite;CAD/CRC;Valuta;0" />');
		document.writeln('<param name="cad_34" value="CADCUC|idc_lite;CAD/CUC;Valuta;0" />');
		document.writeln('<param name="cad_35" value="CADCUP|idc_lite;CAD/CUP;Valuta;0" />');
		document.writeln('<param name="cad_36" value="CADCVE|idc_lite;CAD/CVE;Valuta;0" />');
		document.writeln('<param name="cad_37" value="CADCZK|idc_lite;CAD/CZK;Valuta;0" />');
		document.writeln('<param name="cad_38" value="CADDJF|idc_lite;CAD/DJF;Valuta;0" />');
		document.writeln('<param name="cad_39" value="CADDKK|idc_lite;CAD/DKK;Valuta;0" />');
		document.writeln('<param name="cad_40" value="CADDOP|idc_lite;CAD/DOP;Valuta;0" />');
		document.writeln('<param name="cad_41" value="CADDZD|idc_lite;CAD/DZD;Valuta;0" />');		
		document.writeln('<param name="cad_42" value="CADEGP|idc_lite;CAD/EGP;Valuta;0" />');
		document.writeln('<param name="cad_43" value="CADERN|idc_lite;CAD/ERN;Valuta;0" />');
		document.writeln('<param name="cad_44" value="CADEUR|idc_lite;CAD/EUR;Valuta;0" />');
		document.writeln('<param name="cad_45" value="CADFJD|idc_lite;CAD/FJD;Valuta;0" />');
		document.writeln('<param name="cad_46" value="CADFKP|idc_lite;CAD/FKP;Valuta;0" />');
		document.writeln('<param name="cad_47" value="CADGBP|idc_lite;CAD/GBP;Valuta;0" />');
		document.writeln('<param name="cad_48" value="CADGEL|idc_lite;CAD/GEL;Valuta;0" />');
		document.writeln('<param name="cad_49" value="CADGHS|idc_lite;CAD/GHS;Valuta;0" />');
		document.writeln('<param name="cad_50" value="CADGIP|idc_lite;CAD/GIP;Valuta;0" />');
		document.writeln('<param name="cad_51" value="CADGMD|idc_lite;CAD/GMD;Valuta;0" />');				
		document.writeln('<param name="cad_52" value="CADGNF|idc_lite;CAD/GNF;Valuta;0" />');
		document.writeln('<param name="cad_53" value="CADGTQ|idc_lite;CAD/GTQ;Valuta;0" />');
		document.writeln('<param name="cad_54" value="CADGYD|idc_lite;CAD/GYD;Valuta;0" />');
		document.writeln('<param name="cad_55" value="CADHKD|idc_lite;CAD/HKD;Valuta;0" />');
		document.writeln('<param name="cad_56" value="CADHNL|idc_lite;CAD/HNL;Valuta;0" />');
		document.writeln('<param name="cad_57" value="CADHRK|idc_lite;CAD/HRK;Valuta;0" />');
		document.writeln('<param name="cad_58" value="CADHTG|idc_lite;CAD/HTG;Valuta;0" />');
		document.writeln('<param name="cad_59" value="CADHUF|idc_lite;CAD/HUF;Valuta;0" />');
		document.writeln('<param name="cad_60" value="CADIDR|idc_lite;CAD/IDR;Valuta;0" />');
		document.writeln('<param name="cad_61" value="CADILS|idc_lite;CAD/ILS;Valuta;0" />');		
		document.writeln('<param name="cad_62" value="CADINR|idc_lite;CAD/INR;Valuta;0" />');
		document.writeln('<param name="cad_63" value="CADIQD|idc_lite;CAD/IQD;Valuta;0" />');
		document.writeln('<param name="cad_64" value="CADIRR|idc_lite;CAD/IRR;Valuta;0" />');
		document.writeln('<param name="cad_65" value="CADJMD|idc_lite;CAD/JMD;Valuta;0" />');
		document.writeln('<param name="cad_66" value="CADJOD|idc_lite;CAD/JOD;Valuta;0" />');
		document.writeln('<param name="cad_67" value="CADJPY|idc_lite;CAD/JPY;Valuta;0" />');
		document.writeln('<param name="cad_68" value="CADKES|idc_lite;CAD/KES;Valuta;0" />');
		document.writeln('<param name="cad_69" value="CADKGS|idc_lite;CAD/KGS;Valuta;0" />');
		document.writeln('<param name="cad_70" value="CADKHR|idc_lite;CAD/KHR;Valuta;0" />');
		document.writeln('<param name="cad_71" value="CADKMF|idc_lite;CAD/KMF;Valuta;0" />');				
		document.writeln('<param name="cad_72" value="CADKPW|idc_lite;CAD/KPW;Valuta;0" />');
		document.writeln('<param name="cad_73" value="CADKRW|idc_lite;CAD/KRW;Valuta;0" />');
		document.writeln('<param name="cad_74" value="CADKWD|idc_lite;CAD/KWD;Valuta;0" />');
		document.writeln('<param name="cad_75" value="CADKYD|idc_lite;CAD/KYD;Valuta;0" />');
		document.writeln('<param name="cad_76" value="CADKZT|idc_lite;CAD/KZT;Valuta;0" />');
		document.writeln('<param name="cad_77" value="CADLAK|idc_lite;CAD/LAK;Valuta;0" />');
		document.writeln('<param name="cad_78" value="CADLBP|idc_lite;CAD/LBP;Valuta;0" />');
		document.writeln('<param name="cad_79" value="CADLKR|idc_lite;CAD/LKR;Valuta;0" />');
		document.writeln('<param name="cad_80" value="CADLRD|idc_lite;CAD/LRD;Valuta;0" />');
		document.writeln('<param name="cad_81" value="CADLSL|idc_lite;CAD/LSL;Valuta;0" />');		
		document.writeln('<param name="cad_82" value="CADLTL|idc_lite;CAD/LTL;Valuta;0" />');	
		document.writeln('<param name="cad_83" value="CADLYD|idc_lite;CAD/LYD;Valuta;0" />');
		document.writeln('<param name="cad_84" value="CADMAD|idc_lite;CAD/MAD;Valuta;0" />');
		document.writeln('<param name="cad_85" value="CADMDL|idc_lite;CAD/MDL;Valuta;0" />');
		document.writeln('<param name="cad_86" value="CADMGA|idc_lite;CAD/MGA;Valuta;0" />');
		document.writeln('<param name="cad_87" value="CADMKD|idc_lite;CAD/MKD;Valuta;0" />');
		document.writeln('<param name="cad_88" value="CADMMK|idc_lite;CAD/MMK;Valuta;0" />');
		document.writeln('<param name="cad_89" value="CADMNT|idc_lite;CAD/MNT;Valuta;0" />');
		document.writeln('<param name="cad_90" value="CADMOP|idc_lite;CAD/MOP;Valuta;0" />');				
		document.writeln('<param name="cad_91" value="CADMRO|idc_lite;CAD/MRO;Valuta;0" />');
		document.writeln('<param name="cad_92" value="CADMUR|idc_lite;CAD/MUR;Valuta;0" />');
		document.writeln('<param name="cad_93" value="CADMVR|idc_lite;CAD/MVR;Valuta;0" />');
		document.writeln('<param name="cad_94" value="CADMWK|idc_lite;CAD/MWK;Valuta;0" />');
		document.writeln('<param name="cad_95" value="CADMXN|idc_lite;CAD/MXN;Valuta;0" />');
		document.writeln('<param name="cad_96" value="CADMYR|idc_lite;CAD/MYR;Valuta;0" />');
		document.writeln('<param name="cad_97" value="CADMZN|idc_lite;CAD/MZN;Valuta;0" />');
		document.writeln('<param name="cad_98" value="CADNAD|idc_lite;CAD/NAD;Valuta;0" />');		
		document.writeln('<param name="cad_99" value="CADNGN|idc_lite;CAD/NGN;Valuta;0" />');
		document.writeln('<param name="cad_100" value="CADNIO|idc_lite;CAD/NIO;Valuta;0" />');
		document.writeln('<param name="cad_101" value="CADNOK|idc_lite;CAD/NOK;Valuta;0" />');
		document.writeln('<param name="cad_102" value="CADNPR|idc_lite;CAD/NPR;Valuta;0" />');
		document.writeln('<param name="cad_103" value="CADNZD|idc_lite;CAD/NZD;Valuta;0" />');
		document.writeln('<param name="cad_104" value="CADOMR|idc_lite;CAD/OMR;Valuta;0" />');
		document.writeln('<param name="cad_105" value="CADPAB|idc_lite;CAD/PAB;Valuta;0" />');
		document.writeln('<param name="cad_106" value="CADPEN|idc_lite;CAD/PEN;Valuta;0" />');		
		document.writeln('<param name="cad_107" value="CADPHP|idc_lite;CAD/PHP;Valuta;0" />');
		document.writeln('<param name="cad_108" value="CADPKR|idc_lite;CAD/PKR;Valuta;0" />');
		document.writeln('<param name="cad_109" value="CADPLN|idc_lite;CAD/PLN;Valuta;0" />');
		document.writeln('<param name="cad_110" value="CADPYG|idc_lite;CAD/PYG;Valuta;0" />');
		document.writeln('<param name="cad_111" value="CADQAR|idc_lite;CAD/QAR;Valuta;0" />');
		document.writeln('<param name="cad_112" value="CADRON|idc_lite;CAD/RON;Valuta;0" />');
		document.writeln('<param name="cad_113" value="CADRSD|idc_lite;CAD/RSD;Valuta;0" />');
		document.writeln('<param name="cad_114" value="CADRUB|idc_lite;CAD/RUB;Valuta;0" />');		
		document.writeln('<param name="cad_115" value="CADRWF|idc_lite;CAD/RWF;Valuta;0" />');
		document.writeln('<param name="cad_116" value="CADSAR|idc_lite;CAD/SAR;Valuta;0" />');
		document.writeln('<param name="cad_117" value="CADSCR|idc_lite;CAD/SCR;Valuta;0" />');
		document.writeln('<param name="cad_118" value="CADSDG|idc_lite;CAD/SDG;Valuta;0" />');
		document.writeln('<param name="cad_119" value="CADSEK|idc_lite;CAD/SEK;Valuta;0" />');
		document.writeln('<param name="cad_120" value="CADSGD|idc_lite;CAD/SGD;Valuta;0" />');
		document.writeln('<param name="cad_121" value="CADSHP|idc_lite;CAD/SHP;Valuta;0" />');
		document.writeln('<param name="cad_122" value="CADSLL|idc_lite;CAD/SLL;Valuta;0" />');		
		document.writeln('<param name="cad_123" value="CADSOS|idc_lite;CAD/SOS;Valuta;0" />');
		document.writeln('<param name="cad_124" value="CADSRD|idc_lite;CAD/SRD;Valuta;0" />');
		document.writeln('<param name="cad_125" value="CADSTD|idc_lite;CAD/STD;Valuta;0" />');
		document.writeln('<param name="cad_126" value="CADSVC|idc_lite;CAD/SVC;Valuta;0" />');
		document.writeln('<param name="cad_127" value="CADSYP|idc_lite;CAD/SYP;Valuta;0" />');
		document.writeln('<param name="cad_128" value="CADSZL|idc_lite;CAD/SZL;Valuta;0" />');
		document.writeln('<param name="cad_129" value="CADTHB|idc_lite;CAD/THB;Valuta;0" />');
		document.writeln('<param name="cad_130" value="CADTJS|idc_lite;CAD/TJS;Valuta;0" />');				
		document.writeln('<param name="cad_131" value="CADTMT|idc_lite;CAD/TMT;Valuta;0" />');
		document.writeln('<param name="cad_132" value="CADTND|idc_lite;CAD/TND;Valuta;0" />');
		document.writeln('<param name="cad_133" value="CADTOP|idc_lite;CAD/TOP;Valuta;0" />');
		document.writeln('<param name="cad_134" value="CADTRY|idc_lite;CAD/TRY;Valuta;0" />');		
		document.writeln('<param name="cad_135" value="CADTTD|idc_lite;CAD/TTD;Valuta;0" />');
		document.writeln('<param name="cad_136" value="CADTWD|idc_lite;CAD/TWD;Valuta;0" />');
		document.writeln('<param name="cad_137" value="CADTZS|idc_lite;CAD/TZS;Valuta;0" />');
		document.writeln('<param name="cad_138" value="CADUAH|idc_lite;CAD/UAH;Valuta;0" />');
		document.writeln('<param name="cad_139" value="CADUGX|idc_lite;CAD/UGX;Valuta;0" />');
		document.writeln('<param name="cad_140" value="CADUSD|idc_lite;CAD/USD;Valuta;0" />');
		document.writeln('<param name="cad_141" value="CADUYU|idc_lite;CAD/UYU;Valuta;0" />');
		document.writeln('<param name="cad_142" value="CADVEF|idc_lite;CAD/VEF;Valuta;0" />');		
		document.writeln('<param name="cad_143" value="CADVND|idc_lite;CAD/VND;Valuta;0" />');
		document.writeln('<param name="cad_144" value="CADVUV|idc_lite;CAD/VUV;Valuta;0" />');
		document.writeln('<param name="cad_145" value="CADWST|idc_lite;CAD/WST;Valuta;0" />');
		document.writeln('<param name="cad_146" value="CADXAF|idc_lite;CAD/XAF;Valuta;0" />');
		document.writeln('<param name="cad_147" value="CADXCD|idc_lite;CAD/XCD;Valuta;0" />');
		document.writeln('<param name="cad_148" value="CADXCU|idc_lite;CAD/XCU;Valuta;0" />');
		document.writeln('<param name="cad_149" value="CADXOF|idc_lite;CAD/XOF;Valuta;0" />');
		document.writeln('<param name="cad_150" value="CADXPF|idc_lite;CAD/XPF;Valuta;0" />');			
		document.writeln('<param name="cad_151" value="CADYER|idc_lite;CAD/YER;Valuta;0" />');		
		document.writeln('<param name="cad_152" value="CADZAR|idc_lite;CAD/ZAR;Valuta;0" />');		
		document.writeln('<param name="cad_153" value="CADZMW|idc_lite;CAD/ZMW;Valuta;0" />');		
		document.writeln('<param name="cad_154" value="CADZWL|idc_lite;CAD/ZWL;Valuta;0" />');			
		document.writeln('<param name="iexplore_market_133_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_134" value="@HTML|kyd_|Caymanian Dollar|false|true|false|false" />');
		document.writeln('<param name="kyd_1" value="KYDEUR|idc_lite;KYD/EUR;Valuta;0" />');
		document.writeln('<param name="kyd_2" value="AUDKYD|idc_lite;AUD/KYD;Valuta;0" />');
		document.writeln('<param name="kyd_3" value="CADKYD|idc_lite;CAD/KYD;Valuta;0" />');
		document.writeln('<param name="kyd_4" value="GBPKYD|idc_lite;GBP/KYD;Valuta;0" />');
		document.writeln('<param name="kyd_5" value="JPYKYD|idc_lite;JPY/KYD;Valuta;0" />');
		document.writeln('<param name="kyd_6" value="USDKYD|idc_lite;USD/KYD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_134_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_135" value="@HTML|cve_|Cape Verdean Escudo|false|true|false|false" />');
		document.writeln('<param name="cve_1" value="CVEUSD|idc_lite;CVE/USD;Valuta;0" />');
		document.writeln('<param name="cve_2" value="CADCVE|idc_lite;CHF/CVE;Valuta;0" />');
		document.writeln('<param name="cve_3" value="EURCVE|idc_lite;EUR/CVE;Valuta;0" />');
		document.writeln('<param name="cve_4" value="GBPCVE|idc_lite;GBP/CVE;Valuta;0" />');
		document.writeln('<param name="cve_5" value="JPYCVE|idc_lite;JPY/CVE;Valuta;0" />');
		document.writeln('<param name="cve_6" value="USDCVE|idc_lite;USD/CVE;Valuta;0" />');
		document.writeln('<param name="iexplore_market_135_path" value="Cross Rates" />');		

		document.writeln('<param name="iexplore_market_136" value="@HTML|clp_|Chilean Peso|false|true|false|false" />');
		document.writeln('<param name="clp_1" value="EURCLP|idc_lite;EUR/CLP;Valuta;0" />');
		document.writeln('<param name="clp_2" value="GBPCLP|idc_lite;GBP/CLP;Valuta;0" />');
		document.writeln('<param name="clp_3" value="USDCLP|idc_lite;USD/CLP;Valuta;0" />');
		document.writeln('<param name="clp_4" value="CLPBRL|idc_lite;CLP/BRL;Valuta;0" />');
		document.writeln('<param name="clp_5" value="CLPCAD|idc_lite;CLP/CAD;Valuta;0" />');
		document.writeln('<param name="clp_6" value="CLPCAX|idc_lite;CLP/CAX;Valuta;0" />');		
		document.writeln('<param name="clp_7" value="CLPJPY|idc_lite;CLP/JPY;Valuta;0" />');
		document.writeln('<param name="iexplore_market_136_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_137" value="@HTML|clf_|Chilean Unidad de Fomento|false|true|false|false" />');
		document.writeln('<param name="clf_1" value="CLFCLP|idc_lite;CLF/CLP;Valuta;0" />');
		document.writeln('<param name="clf_2" value="CLFUSD|idc_lite;CLF/USD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_137_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_138" value="@HTML|cnh_|Chinese Offshore Yuan|false|true|false|false" />');
		document.writeln('<param name="cnh_1" value="CNHHKD|idc_lite;CNH/HKD;Valuta;0" />');
		document.writeln('<param name="cnh_2" value="CNHIDR|idc_lite;CNH/IDR;Valuta;0" />');
		document.writeln('<param name="cnh_3" value="CNHINR|idc_lite;CNH/INR;Valuta;0" />');
		document.writeln('<param name="cnh_4" value="CNHJPY|idc_lite;CNH/JPY;Valuta;0" />');
		document.writeln('<param name="cnh_5" value="CNHKRW|idc_lite;CNH/KRW;Valuta;0" />');
		document.writeln('<param name="cnh_6" value="CNHPHP|idc_lite;CNH/PHP;Valuta;0" />');
		document.writeln('<param name="cnh_7" value="CNHTHB|idc_lite;CNH/THB;Valuta;0" />');
		document.writeln('<param name="cnh_8" value="CNHTWD|idc_lite;CNH/TWD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_138_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_139" value="@HTML|cny_|Chinese Yuan|false|true|false|false" />');
		document.writeln('<param name="cny_1" value="EURCNY|idc_lite;EUR/CNY;Valuta;0" />');
		document.writeln('<param name="cny_2" value="GBPCNY|idc_lite;GBP/CNY;Valuta;0" />');
		document.writeln('<param name="cny_3" value="JPYCNY|idc_lite;JPY/CNY;Valuta;0" />');
		document.writeln('<param name="cny_4" value="USDCNY|idc_lite;USD/CNY;Valuta;0" />');
		document.writeln('<param name="cny_5" value="CNYAUD|idc_lite;CNY/AUD;Valuta;0" />');
		document.writeln('<param name="cny_6" value="CNYBRL|idc_lite;CNY/BRL;Valuta;0" />');
		document.writeln('<param name="cny_7" value="CNYCAD|idc_lite;CNY/CAD;Valuta;0" />');		
		document.writeln('<param name="cny_8" value="CNYCHF|idc_lite;CNY/CHF;Valuta;0" />');
		document.writeln('<param name="cny_9" value="CNYEUR|idc_lite;CNY/EUR;Valuta;0" />');
		document.writeln('<param name="cny_10" value="CNYGBP|idc_lite;CNY/GBP;Valuta;0" />');			
		document.writeln('<param name="cny_11" value="CNYHKD|idc_lite;CNY/HKD;Valuta;0" />');
		document.writeln('<param name="cny_12" value="CNYIDR|idc_lite;CNY/IDR;Valuta;0" />');
		document.writeln('<param name="cny_13" value="CNYINR|idc_lite;CNY/INR;Valuta;0" />');			
		document.writeln('<param name="cny_14" value="CNYJPY|idc_lite;CNY/JPY;Valuta;0" />');
		document.writeln('<param name="cny_15" value="CNYKRW|idc_lite;CNY/KRW;Valuta;0" />');
		document.writeln('<param name="cny_16" value="CNYMYR|idc_lite;CNY/MYR;Valuta;0" />');			
		document.writeln('<param name="cny_17" value="CNYNZD|idc_lite;CNY/NZD;Valuta;0" />');
		document.writeln('<param name="cny_18" value="CNYPHP|idc_lite;CNY/PHP;Valuta;0" />');
		document.writeln('<param name="cny_19" value="CNYSEK|idc_lite;CNY/SEK;Valuta;0" />');			
		document.writeln('<param name="cny_20" value="CNYSGD|idc_lite;CNY/SGD;Valuta;0" />');
		document.writeln('<param name="cny_21" value="CNYTHB|idc_lite;CNY/THB;Valuta;0" />');
		document.writeln('<param name="cny_22" value="CNYTWD|idc_lite;CNY/TWD;Valuta;0" />');		
		document.writeln('<param name="cny_23" value="CNYUSD|idc_lite;CNY/USD;Valuta;0" />');
		document.writeln('<param name="cny_24" value="CNYXCU|idc_lite;CNY/XCU;Valuta;0" />');			
		document.writeln('<param name="cny_25" value="CNYZAR|idc_lite;CNY/ZAR;Valuta;0" />');		
		document.writeln('<param name="iexplore_market_139_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_140" value="@HTML|cop_|Colombian Peso|false|true|false|false" />');
		document.writeln('<param name="cop_1" value="EURCOP|idc_lite;EUR/COP;Valuta;0" />');
		document.writeln('<param name="cop_2" value="GBPCOP|idc_lite;GBP/COP;Valuta;0" />');
		document.writeln('<param name="cop_3" value="USDCOP|idc_lite;USD/COP;Valuta;0" />');
		document.writeln('<param name="cop_4" value="COPBRL|idc_lite;COP/BRL;Valuta;0" />');		
		document.writeln('<param name="cop_5" value="COPEUR|idc_lite;COP/EUR;Valuta;0" />');
		document.writeln('<param name="cop_6" value="COPJPY|idc_lite;COP/JPY;Valuta;0" />');		
		document.writeln('<param name="iexplore_market_140_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_141" value="@HTML|cuc_|Cuban Convertible Peso|false|true|false|false" />');
		document.writeln('<param name="cuc_1" value="CUCJPY|idc_lite;CUC/JPY;Valuta;0" />');
		document.writeln('<param name="cuc_2" value="CADCUC|idc_lite;CAD/CUC;Valuta;0" />');
		document.writeln('<param name="cuc_3" value="CHFCUC|idc_lite;CHF/CUC;Valuta;0" />');
		document.writeln('<param name="cuc_4" value="DKKCUC|idc_lite;DKK/CUC;Valuta;0" />');
		document.writeln('<param name="cuc_5" value="EURCUC|idc_lite;EUR/CUC;Valuta;0" />');
		document.writeln('<param name="cuc_6" value="GBPCUC|idc_lite;GBP/CUC;Valuta;0" />');
		document.writeln('<param name="cuc_7" value="MXNCUC|idc_lite;MXN/CUC;Valuta;0" />');
		document.writeln('<param name="cuc_8" value="NOKCUC|idc_lite;NOK/CUC;Valuta;0" />');	
		document.writeln('<param name="cuc_9" value="SEKCUC|idc_lite;SEK/CUC;Valuta;0" />');
		document.writeln('<param name="cuc_10" value="USDCUC|idc_lite;USD/CUC;Valuta;0" />');
		document.writeln('<param name="iexplore_market_141_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_142" value="@HTML|cup_|Cuban Peso|false|true|false|false" />');
		document.writeln('<param name="cup_1" value="CUPUSD|idc_lite;CUP/USD;Valuta;0" />');
		document.writeln('<param name="cup_2" value="CADCUP|idc_lite;CAD/CUP;Valuta;0" />');
		document.writeln('<param name="cup_3" value="GBPCUP|idc_lite;GBP/CUP;Valuta;0" />');
		document.writeln('<param name="cup_4" value="JPYCUP|idc_lite;JPY/CUP;Valuta;0" />');
		document.writeln('<param name="cup_5" value="USDCUP|idc_lite;USD/CUP;Valuta;0" />');
		document.writeln('<param name="iexplore_market_142_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_143" value="@HTML|hrk_|Croatian Kuna|false|true|false|false" />');
		document.writeln('<param name="hrk_1" value="EURHRK|idc_lite;EUR/HRK;Valuta;0" />');
		document.writeln('<param name="hrk_2" value="GBPHRK|idc_lite;GBP/HRK;Valuta;0" />');
		document.writeln('<param name="hrk_3" value="USDHRK|idc_lite;USD/HRK;Valuta;0" />');
		document.writeln('<param name="hrk_4" value="HRKCHF|idc_lite;HRK/CHF;Valuta;0" />');
		document.writeln('<param name="hrk_5" value="HRKEUR|idc_lite;HRK/EUR;Valuta;0" />');
		document.writeln('<param name="hrk_6" value="AUDHRK|idc_lite;AUD/HRK;Valuta;0" />');
		document.writeln('<param name="hrk_7" value="CADHRK|idc_lite;CAD/HRK;Valuta;0" />');
		document.writeln('<param name="hrk_8" value="CHFHRK|idc_lite;CHF/HRK;Valuta;0" />');
		document.writeln('<param name="hrk_9" value="EUXHRK|idc_lite;EUX/HRK;Valuta;0" />');
		document.writeln('<param name="hrk_10" value="JPYHRK|idc_lite;JPY/HRK;Valuta;0" />');
		document.writeln('<param name="iexplore_market_143_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_144" value="@HTML|czk_|Czech Koruna|false|true|false|false" />');
		document.writeln('<param name="czk_1" value="CHFCZK|idc_lite;CHF/CZK;Valuta;0" />');
		document.writeln('<param name="czk_2" value="EURCZK|idc_lite;EUR/CZK;Valuta;0" />');
		document.writeln('<param name="czk_3" value="GBPCZK|idc_lite;GBP/CZK;Valuta;0" />');
		document.writeln('<param name="czk_4" value="USDCZK|idc_lite;USD/CZK;Valuta;0" />');
		document.writeln('<param name="czk_5" value="CZKAUD|idc_lite;CZK/AUD;Valuta;0" />');
		document.writeln('<param name="czk_6" value="CZKCAD|idc_lite;CZK/CAD;Valuta;0" />');
		document.writeln('<param name="czk_7" value="CZKCAX|idc_lite;CZK/CAX;Valuta;0" />');
		document.writeln('<param name="czk_8" value="CZKCHF|idc_lite;CZK/CHF;Valuta;0" />');
		document.writeln('<param name="czk_9" value="CZKDKK|idc_lite;CZK/DKK;Valuta;0" />');
		document.writeln('<param name="czk_10" value="CZKEUR|idc_lite;CZK/EUR;Valuta;0" />');
		document.writeln('<param name="czk_11" value="CZKGBP|idc_lite;CZK/GBP;Valuta;0" />');
		document.writeln('<param name="czk_12" value="CZKHKD|idc_lite;CZK/HKD;Valuta;0" />');
		document.writeln('<param name="czk_13" value="CZKHUF|idc_lite;CZK/HUF;Valuta;0" />');
		document.writeln('<param name="czk_14" value="CZKJPY|idc_lite;CZK/JPY;Valuta;0" />');
		document.writeln('<param name="czk_15" value="CZKNOK|idc_lite;CZK/NOK;Valuta;0" />');
		document.writeln('<param name="czk_16" value="CZKPLN|idc_lite;CZK/PLN;Valuta;0" />');
		document.writeln('<param name="czk_17" value="CZKSEK|idc_lite;CZK/SEK;Valuta;0" />');
		document.writeln('<param name="czk_18" value="CZKUSD|idc_lite;CZK/USD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_144_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_145" value="@HTML|dkk_|Danish Krone|false|true|false|false" />');
		document.writeln('<param name="dkk_1" value="CHFDKK|idc_lite;CHF/DKK;Valuta;0" />');
		document.writeln('<param name="dkk_2" value="EURDKK|idc_lite;EUR/DKK;Valuta;0" />');
		document.writeln('<param name="dkk_3" value="GBPDKK|idc_lite;GBP/DKK;Valuta;0" />');
		document.writeln('<param name="dkk_4" value="JPYDKK|idc_lite;JPY/DKK;Valuta;0" />');
		document.writeln('<param name="dkk_5" value="USDDKK|idc_lite;USD/DKK;Valuta;0" />');
		document.writeln('<param name="dkk_6" value="DKKANG|idc_lite;DKK/ANG;Valuta;0" />');
		document.writeln('<param name="dkk_7" value="DKKAUD|idc_lite;DKK/AUD;Valuta;0" />');
		document.writeln('<param name="dkk_8" value="DKKBAM|idc_lite;DKK/BAM;Valuta;0" />');
		document.writeln('<param name="dkk_9" value="DKKBRL|idc_lite;DKK/BRL;Valuta;0" />');
		document.writeln('<param name="dkk_10" value="DKKCAD|idc_lite;DKK/CAD;Valuta;0" />');
		document.writeln('<param name="dkk_11" value="DKKCHF|idc_lite;DKK/CHF;Valuta;0" />');
		document.writeln('<param name="dkk_12" value="DKKCUC|idc_lite;DKK/CUC;Valuta;0" />');
		document.writeln('<param name="dkk_13" value="DKKCZK|idc_lite;DKK/CZK;Valuta;0" />');
		document.writeln('<param name="dkk_14" value="DKKEUR|idc_lite;DKK/EUR;Valuta;0" />');
		document.writeln('<param name="dkk_15" value="DKKGBP|idc_lite;DKK/GBP;Valuta;0" />');
		document.writeln('<param name="dkk_16" value="DKKGIP|idc_lite;DKK/GIP;Valuta;0" />');
		document.writeln('<param name="dkk_17" value="DKKHKD|idc_lite;DKK/HKD;Valuta;0" />');		
		document.writeln('<param name="dkk_18" value="DKKHUF|idc_lite;DKK/HUF;Valuta;0" />');
		document.writeln('<param name="dkk_19" value="DKKIDR|idc_lite;DKK/IDR;Valuta;0" />');
		document.writeln('<param name="dkk_20" value="DKKINR|idc_lite;DKK/INR;Valuta;0" />');
		document.writeln('<param name="dkk_21" value="DKKJPY|idc_lite;DKK/JPY;Valuta;0" />');
		document.writeln('<param name="dkk_22" value="DKKKPW|idc_lite;DKK/KPW;Valuta;0" />');
		document.writeln('<param name="dkk_23" value="DKKKRW|idc_lite;DKK/KRW;Valuta;0" />');		
		document.writeln('<param name="dkk_25" value="DKKNOK|idc_lite;DKK/NOK;Valuta;0" />');
		document.writeln('<param name="dkk_26" value="DKKNZD|idc_lite;DKK/NZD;Valuta;0" />');
		document.writeln('<param name="dkk_27" value="DKKPKR|idc_lite;DKK/PKR;Valuta;0" />');
		document.writeln('<param name="dkk_28" value="DKKPLN|idc_lite;DKK/PLN;Valuta;0" />');
		document.writeln('<param name="dkk_29" value="DKKQAR|idc_lite;DKK/QAR;Valuta;0" />');			
		document.writeln('<param name="dkk_30" value="DKKRON|idc_lite;DKK/RON;Valuta;0" />');
		document.writeln('<param name="dkk_31" value="DKKRSD|idc_lite;DKK/RSD;Valuta;0" />');
		document.writeln('<param name="dkk_32" value="DKKSAR|idc_lite;DKK/SAR;Valuta;0" />');
		document.writeln('<param name="dkk_33" value="DKKSEK|idc_lite;DKK/SEK;Valuta;0" />');
		document.writeln('<param name="dkk_34" value="DKKSGD|idc_lite;DKK/SGD;Valuta;0" />');
		document.writeln('<param name="dkk_35" value="DKKTHB|idc_lite;DKK/THB;Valuta;0" />');
		document.writeln('<param name="dkk_36" value="DKKUSD|idc_lite;DKK/USD;Valuta;0" />');
		document.writeln('<param name="dkk_37" value="DKKXCU|idc_lite;DKK/XCU;Valuta;0" />');
		document.writeln('<param name="dkk_38" value="DKKZAR|idc_lite;DKK/ZAR;Valuta;0" />');	
		document.writeln('<param name="iexplore_market_145_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_146" value="@HTML|djf_|Djiboutian Franc|false|true|false|false" />');
		document.writeln('<param name="djf_1" value="DJFGBP|idc_lite;DJF/GBP;Valuta;0" />');
		document.writeln('<param name="djf_2" value="DJFUSD|idc_lite;DJF/USD;Valuta;0" />');
		document.writeln('<param name="djf_3" value="CADDJF|idc_lite;CAD/DJF;Valuta;0" />');
		document.writeln('<param name="djf_4" value="CHFDJF|idc_lite;CHF/DJF;Valuta;0" />');
		document.writeln('<param name="djf_5" value="EURDJF|idc_lite;EUR/DJF;Valuta;0" />');
		document.writeln('<param name="djf_6" value="GBPDJF|idc_lite;GBP/DJF;Valuta;0" />');
		document.writeln('<param name="djf_7" value="JPYDJF|idc_lite;JPY/DJF;Valuta;0" />');
		document.writeln('<param name="djf_8" value="USDDJF|idc_lite;USD/DJF;Valuta;0" />');
		document.writeln('<param name="iexplore_market_146_path" value="Cross Rates" />');
				
		document.writeln('<param name="iexplore_market_147" value="@HTML|egp_|Egyptian Pound|false|true|false|false" />');
		document.writeln('<param name="egp_1" value="EUREGP|idc_lite;EUR/EGP;Valuta;0" />');
		document.writeln('<param name="egp_2" value="GBPEGP|idc_lite;GBP/EGP;Valuta;0" />');
		document.writeln('<param name="egp_3" value="USDEGP|idc_lite;USD/EGP;Valuta;0" />');
		document.writeln('<param name="egp_4" value="EGPCHF|idc_lite;EGP/CHF;Valuta;0" />');
		document.writeln('<param name="egp_5" value="EGPEUR|idc_lite;EGP/EUR;Valuta;0" />');
		document.writeln('<param name="egp_6" value="EGPGBP|idc_lite;EGP/GBP;Valuta;0" />');
		document.writeln('<param name="egp_7" value="EGPPKR|idc_lite;EGP/PKR;Valuta;0" />');
		document.writeln('<param name="egp_8" value="EGPUSD|idc_lite;EGP/USD;Valuta;0" />');
		document.writeln('<param name="egp_9" value="EGPZAR|idc_lite;EGP/ZAR;Valuta;0" />');		
		document.writeln('<param name="iexplore_market_147_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_148" value="@HTML|xcd_|East Caribbean Dollar|false|true|false|false" />');
		document.writeln('<param name="xcd_1" value="XCDGBP|idc_lite;XCD/GBP;Valuta;0" />');
		document.writeln('<param name="xcd_2" value="AUDXCD|idc_lite;AUD/XCD;Valuta;0" />');
		document.writeln('<param name="xcd_3" value="CADXCD|idc_lite;CAD/XCD;Valuta;0" />');
		document.writeln('<param name="xcd_4" value="GBPXCD|idc_lite;GBP/XCD;Valuta;0" />');
		document.writeln('<param name="xcd_5" value="JPYXCD|idc_lite;JPY/XCD;Valuta;0" />');
		document.writeln('<param name="xcd_6" value="USDXCD|idc_lite;USD/XCD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_148_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_149" value="@HTML|eur_|Euro|false|true|false|false" />');
		document.writeln('<param name="eur_1" value="EURAED|idc_lite;EUR/AED;Valuta;0" />');
		document.writeln('<param name="eur_2" value="EURAFN|idc_lite;EUR/AFN;Valuta;0" />');
		document.writeln('<param name="eur_3" value="EURALL|idc_lite;EUR/ALL;Valuta;0" />');		
		document.writeln('<param name="eur_4" value="EURANG|idc_lite;EUR/ANG;Valuta;0" />');		
		document.writeln('<param name="eur_5" value="EURARS|idc_lite;EUR/ARS;Valuta;0" />');
		document.writeln('<param name="eur_6" value="EURAUD|idc_lite;EUR/AUD;Valuta;0" />');
		document.writeln('<param name="eur_7" value="EURAWG|idc_lite;EUR/AWG;Valuta;0" />');	
		document.writeln('<param name="eur_8" value="EURBAM|idc_lite;EUR/BAM;Valuta;0" />');
		document.writeln('<param name="eur_9" value="EURBDT|idc_lite;EUR/BDT;Valuta;0" />');		
		document.writeln('<param name="eur_10" value="EURBGN|idc_lite;EUR/BGN;Valuta;0" />');
		document.writeln('<param name="eur_11" value="EURBHD|idc_lite;EUR/BHD;Valuta;0" />');
		document.writeln('<param name="eur_12" value="EURBIF|idc_lite;EUR/BIF;Valuta;0" />');		
		document.writeln('<param name="eur_13" value="EURBND|idc_lite;EUR/BND;Valuta;0" />');		
		document.writeln('<param name="eur_14" value="EURBOB|idc_lite;EUR/BOB;Valuta;0" />');		
		document.writeln('<param name="eur_15" value="EURBRL|idc_lite;EUR/BRL;Valuta;0" />');
		document.writeln('<param name="eur_16" value="EURBSD|idc_lite;EUR/BSD;Valuta;0" />');		
		document.writeln('<param name="eur_17" value="EURBWP|idc_lite;EUR/BWP;Valuta;0" />');				
		document.writeln('<param name="eur_18" value="EURBYR|idc_lite;EUR/BYR;Valuta;0" />');				
		document.writeln('<param name="eur_19" value="EURBZD|idc_lite;EUR/BZD;Valuta;0" />');			
		document.writeln('<param name="eur_20" value="EURCAD|idc_lite;EUR/CAD;Valuta;0" />');
		document.writeln('<param name="eur_21" value="EURCDF|idc_lite;EUR/CDF;Valuta;0" />');
		document.writeln('<param name="eur_22" value="EURCHF|idc_lite;EUR/CHF;Valuta;0" />');
		document.writeln('<param name="eur_23" value="EURCLP|idc_lite;EUR/CLP;Valuta;0" />');
		document.writeln('<param name="eur_23" value="EURCNH|idc_lite;EUR/CNH;Valuta;0" />');
		document.writeln('<param name="eur_24" value="EURCNY|idc_lite;EUR/CNY;Valuta;0" />');
		document.writeln('<param name="eur_25" value="EURCOP|idc_lite;EUR/COP;Valuta;0" />');
		document.writeln('<param name="eur_26" value="EURCRC|idc_lite;EUR/CRC;Valuta;0" />');
		document.writeln('<param name="eur_27" value="EURCUC|idc_lite;EUR/CUC;Valuta;0" />');
		document.writeln('<param name="eur_28" value="EURCVE|idc_lite;EUR/CVE;Valuta;0" />');		
		document.writeln('<param name="eur_29" value="EURCZK|idc_lite;EUR/CZK;Valuta;0" />');		
		document.writeln('<param name="eur_30" value="EURDJF|idc_lite;EUR/DJF;Valuta;0" />');
		document.writeln('<param name="eur_31" value="EURDKK|idc_lite;EUR/DKK;Valuta;0" />');
		document.writeln('<param name="eur_32" value="EURDOP|idc_lite;EUR/DOP;Valuta;0" />');
		document.writeln('<param name="eur_33" value="EURDZD|idc_lite;EUR/DZD;Valuta;0" />');	
		document.writeln('<param name="eur_34" value="EUREGP|idc_lite;EUR/EGP;Valuta;0" />');
		document.writeln('<param name="eur_35" value="EURERN|idc_lite;EUR/ERN;Valuta;0" />');
		document.writeln('<param name="eur_36" value="EURFKP|idc_lite;EUR/FKP;Valuta;0" />');	
		document.writeln('<param name="eur_37" value="EURGBP|idc_lite;EUR/GBP;Valuta;0" />');		
		document.writeln('<param name="eur_38" value="EURGEL|idc_lite;EUR/GEL;Valuta;0" />');		
		document.writeln('<param name="eur_39" value="EURGHS|idc_lite;EUR/GHS;Valuta;0" />');			
		document.writeln('<param name="eur_40" value="EURGIP|idc_lite;EUR/GIP;Valuta;0" />');		
		document.writeln('<param name="eur_41" value="EURGMD|idc_lite;EUR/GMD;Valuta;0" />');		
		document.writeln('<param name="eur_42" value="EURGNF|idc_lite;EUR/GNF;Valuta;0" />');			
		document.writeln('<param name="eur_43" value="EURGTQ|idc_lite;EUR/GTQ;Valuta;0" />');	
		document.writeln('<param name="eur_44" value="EURGYD|idc_lite;EUR/GYD;Valuta;0" />');
		document.writeln('<param name="eur_45" value="EURHKD|idc_lite;EUR/HKD;Valuta;0" />');
		document.writeln('<param name="eur_46" value="EURHNL|idc_lite;EUR/HNL;Valuta;0" />');
		document.writeln('<param name="eur_47" value="EURHRK|idc_lite;EUR/HRK;Valuta;0" />');		
		document.writeln('<param name="eur_48" value="EURHTG|idc_lite;EUR/HTG;Valuta;0" />');
		document.writeln('<param name="eur_49" value="EURHUF|idc_lite;EUR/HUF;Valuta;0" />');
		document.writeln('<param name="eur_50" value="EURIDR|idc_lite;EUR/IDR;Valuta;0" />');
		document.writeln('<param name="eur_51" value="EURILS|idc_lite;EUR/ILS;Valuta;0" />');
		document.writeln('<param name="eur_52" value="EURINR|idc_lite;EUR/INR;Valuta;0" />');
		document.writeln('<param name="eur_53" value="EURIQD|idc_lite;EUR/IQD;Valuta;0" />');
		document.writeln('<param name="eur_54" value="EURIRR|idc_lite;EUR/IRR;Valuta;0" />');
		document.writeln('<param name="eur_55" value="EURJMD|idc_lite;EUR/JMD;Valuta;0" />');
		document.writeln('<param name="eur_56" value="EURJOD|idc_lite;EUR/JOD;Valuta;0" />');
		document.writeln('<param name="eur_57" value="EURJPY|idc_lite;EUR/JPY;Valuta;0" />');
		document.writeln('<param name="eur_58" value="EURKES|idc_lite;EUR/KES;Valuta;0" />');		
		document.writeln('<param name="eur_59" value="EURKHR|idc_lite;EUR/KHR;Valuta;0" />');
		document.writeln('<param name="eur_60" value="EURKMF|idc_lite;EUR/KMF;Valuta;0" />');
		document.writeln('<param name="eur_61" value="EURKPW|idc_lite;EUR/KPW;Valuta;0" />');				
		document.writeln('<param name="eur_62" value="EURKRW|idc_lite;EUR/KRW;Valuta;0" />');
		document.writeln('<param name="eur_63" value="EURKWD|idc_lite;EUR/KWD;Valuta;0" />');
		document.writeln('<param name="eur_64" value="EURKZT|idc_lite;EUR/KZT;Valuta;0" />');	
		document.writeln('<param name="eur_65" value="EURLAK|idc_lite;EUR/LAK;Valuta;0" />');	
		document.writeln('<param name="eur_66" value="EURLBP|idc_lite;EUR/LBP;Valuta;0" />');	
		document.writeln('<param name="eur_67" value="EURLKR|idc_lite;EUR/LKR;Valuta;0" />');	
		document.writeln('<param name="eur_68" value="EURLRD|idc_lite;EUR/LRD;Valuta;0" />');	
		document.writeln('<param name="eur_69" value="EURLSL|idc_lite;EUR/LSL;Valuta;0" />');		
		document.writeln('<param name="eur_70" value="EURLTL|idc_lite;EUR/LTL;Valuta;0" />');	
		document.writeln('<param name="eur_71" value="EURLYD|idc_lite;EUR/LYD;Valuta;0" />');	
		document.writeln('<param name="eur_72" value="EURMAD|idc_lite;EUR/MAD;Valuta;0" />');
		document.writeln('<param name="eur_73" value="EURMDL|idc_lite;EUR/MDL;Valuta;0" />');	
		document.writeln('<param name="eur_74" value="EURMGA|idc_lite;EUR/MGA;Valuta;0" />');
		document.writeln('<param name="eur_75" value="EURMKD|idc_lite;EUR/MKD;Valuta;0" />');	
		document.writeln('<param name="eur_76" value="EURMMK|idc_lite;EUR/MMK;Valuta;0" />');
		document.writeln('<param name="eur_77" value="EURMNT|idc_lite;EUR/MNT;Valuta;0" />');	
		document.writeln('<param name="eur_78" value="EURMRO|idc_lite;EUR/MRO;Valuta;0" />');
		document.writeln('<param name="eur_79" value="EURMUR|idc_lite;EUR/MUR;Valuta;0" />');	
		document.writeln('<param name="eur_80" value="EURMVR|idc_lite;EUR/MVR;Valuta;0" />');
		document.writeln('<param name="eur_81" value="EURMWK|idc_lite;EUR/MWK;Valuta;0" />');	
		document.writeln('<param name="eur_82" value="EURMXN|idc_lite;EUR/MXN;Valuta;0" />');
		document.writeln('<param name="eur_83" value="EURMYR|idc_lite;EUR/MYR;Valuta;0" />');
		document.writeln('<param name="eur_84" value="EURMZN|idc_lite;EUR/MZN;Valuta;0" />');		
		document.writeln('<param name="eur_85" value="EURNAD|idc_lite;EUR/NAD;Valuta;0" />');	
		document.writeln('<param name="eur_86" value="EURNGN|idc_lite;EUR/NGN;Valuta;0" />');		
		document.writeln('<param name="eur_87" value="EURNIO|idc_lite;EUR/NIO;Valuta;0" />');
		document.writeln('<param name="eur_88" value="EURNOK|idc_lite;EUR/NOK;Valuta;0" />');		
		document.writeln('<param name="eur_89" value="EURNPR|idc_lite;EUR/NPR;Valuta;0" />');
		document.writeln('<param name="eur_90" value="EURNZD|idc_lite;EUR/NZD;Valuta;0" />');
		document.writeln('<param name="eur_91" value="EUROMR|idc_lite;EUR/OMR;Valuta;0" />');		
		document.writeln('<param name="eur_92" value="EURPEN|idc_lite;EUR/PEN;Valuta;0" />');				
		document.writeln('<param name="eur_93" value="EURPGK|idc_lite;EUR/PGK;Valuta;0" />');				
		document.writeln('<param name="eur_94" value="EURPHP|idc_lite;EUR/PHP;Valuta;0" />');
		document.writeln('<param name="eur_95" value="EURPKR|idc_lite;EUR/PKR;Valuta;0" />');
		document.writeln('<param name="eur_96" value="EURPLN|idc_lite;EUR/PLN;Valuta;0" />');
		document.writeln('<param name="eur_97" value="EURPYG|idc_lite;EUR/PYG;Valuta;0" />');
		document.writeln('<param name="eur_98" value="EURQAR|idc_lite;EUR/QAR;Valuta;0" />');		
		document.writeln('<param name="eur_99" value="EURRON|idc_lite;EUR/RON;Valuta;0" />');
		document.writeln('<param name="eur_100" value="EURRSD|idc_lite;EUR/RSD;Valuta;0" />');
		document.writeln('<param name="eur_101" value="EURRUB|idc_lite;EUR/RUB;Valuta;0" />');
		document.writeln('<param name="eur_102" value="EURRWF|idc_lite;EUR/RWF;Valuta;0" />');
		document.writeln('<param name="eur_103" value="EURSAR|idc_lite;EUR/SAR;Valuta;0" />');		
		document.writeln('<param name="eur_104" value="EURSCR|idc_lite;EUR/SCR;Valuta;0" />');
		document.writeln('<param name="eur_105" value="EURSDG|idc_lite;EUR/SDG;Valuta;0" />');	
		document.writeln('<param name="eur_106" value="EURSEK|idc_lite;EUR/SEK;Valuta;0" />');
		document.writeln('<param name="eur_107" value="EURSGD|idc_lite;EUR/SGD;Valuta;0" />');
		document.writeln('<param name="eur_108" value="EURSHP|idc_lite;EUR/SHP;Valuta;0" />');		
		document.writeln('<param name="eur_109" value="EURSLL|idc_lite;EUR/SLL;Valuta;0" />');	
		document.writeln('<param name="eur_110" value="EURSOS|idc_lite;EUR/SOS;Valuta;0" />');
		document.writeln('<param name="eur_111" value="EURSRD|idc_lite;EUR/SRD;Valuta;0" />');		
		document.writeln('<param name="eur_112" value="EURSTD|idc_lite;EUR/STD;Valuta;0" />');	
		document.writeln('<param name="eur_113" value="EURSVC|idc_lite;EUR/SVC;Valuta;0" />');
		document.writeln('<param name="eur_114" value="EURSYP|idc_lite;EUR/SYP;Valuta;0" />');		
		document.writeln('<param name="eur_115" value="EURSZL|idc_lite;EUR/SZL;Valuta;0" />');		
		document.writeln('<param name="eur_116" value="EURTHB|idc_lite;EUR/THB;Valuta;0" />');
		document.writeln('<param name="eur_117" value="EURTMT|idc_lite;EUR/TMT;Valuta;0" />');
		document.writeln('<param name="eur_118" value="EURTND|idc_lite;EUR/TND;Valuta;0" />');
		document.writeln('<param name="eur_119" value="EURTOP|idc_lite;EUR/TOP;Valuta;0" />');
		document.writeln('<param name="eur_120" value="EURTRY|idc_lite;EUR/TRY;Valuta;0" />');
		document.writeln('<param name="eur_121" value="EURTTD|idc_lite;EUR/TTD;Valuta;0" />');
		document.writeln('<param name="eur_122" value="EURTWD|idc_lite;EUR/TWD;Valuta;0" />');
		document.writeln('<param name="eur_123" value="EURTZS|idc_lite;EUR/TZS;Valuta;0" />');		
		document.writeln('<param name="eur_124" value="EURUAH|idc_lite;EUR/UAH;Valuta;0" />');
		document.writeln('<param name="eur_125" value="EURUGX|idc_lite;EUR/UGX;Valuta;0" />');
		document.writeln('<param name="eur_126" value="EURUSD|idc_lite;EUR/USD;Valuta;0" />');	
		document.writeln('<param name="eur_127" value="EURUYU|idc_lite;EUR/UYU;Valuta;0" />');
		document.writeln('<param name="eur_128" value="EURVEF|idc_lite;EUR/VEF;Valuta;0" />');
		document.writeln('<param name="eur_129" value="EURVND|idc_lite;EUR/VND;Valuta;0" />');
		document.writeln('<param name="eur_130" value="EURVUV|idc_lite;EUR/VUV;Valuta;0" />');
		document.writeln('<param name="eur_131" value="EURWST|idc_lite;EUR/WST;Valuta;0" />');
		document.writeln('<param name="eur_132" value="EURXAF|idc_lite;EUR/XAF;Valuta;0" />');
		document.writeln('<param name="eur_133" value="EURXCU|idc_lite;EUR/XCU;Valuta;0" />');
		document.writeln('<param name="eur_134" value="EURXOF|idc_lite;EUR/XOF;Valuta;0" />');
		document.writeln('<param name="eur_135" value="EURXPF|idc_lite;EUR/XPF;Valuta;0" />');
		document.writeln('<param name="eur_136" value="EURYER|idc_lite;EUR/YER;Valuta;0" />');
		document.writeln('<param name="eur_137" value="EURZAR|idc_lite;EUR/ZAR;Valuta;0" />');
		document.writeln('<param name="eur_138" value="EURZMW|idc_lite;EUR/ZMW;Valuta;0" />');	
		document.writeln('<param name="eur_139" value="EURZWL|idc_lite;EUR/ZWL;Valuta;0" />');	
		document.writeln('<param name="iexplore_market_149_path" value="Cross Rates" />');	
		
		document.writeln('<param name="iexplore_market_150" value="@HTML|eux_|Euro Reference Rates|false|true|false|false" />');
		document.writeln('<param name="eux_1" value="EUXBGN|idc_lite;EUX/BGN;Valuta;0" />');
		document.writeln('<param name="eux_2" value="EUXBRL|idc_lite;EUX/BRL;Valuta;0" />');
		document.writeln('<param name="eux_3" value="EUXCAD|idc_lite;EUX/CAD;Valuta;0" />');
		document.writeln('<param name="eux_4" value="EUXCHF|idc_lite;EUX/CHF;Valuta;0" />');
		document.writeln('<param name="eux_5" value="EUXCNY|idc_lite;EUX/CNY;Valuta;0" />');
		document.writeln('<param name="eux_6" value="EUXCZK|idc_lite;EUX/CZK;Valuta;0" />');
		document.writeln('<param name="eux_7" value="EUXDKK|idc_lite;EUX/DKK;Valuta;0" />');
		document.writeln('<param name="eux_8" value="EUXGBP|idc_lite;EUX/GBP;Valuta;0" />');
		document.writeln('<param name="eux_9" value="EUXHKD|idc_lite;EUX/HKD;Valuta;0" />');
		document.writeln('<param name="eux_10" value="EUXHRK|idc_lite;EUX/HRK;Valuta;0" />');
		document.writeln('<param name="eux_11" value="EUXIDR|idc_lite;EUX/IDR;Valuta;0" />');
		document.writeln('<param name="eux_12" value="EUXILS|idc_lite;EUX/ILS;Valuta;0" />');
		document.writeln('<param name="eux_13" value="EUXINR|idc_lite;EUX/INR;Valuta;0" />');
		document.writeln('<param name="eux_14" value="EUXKRW|idc_lite;EUX/KRW;Valuta;0" />');
		document.writeln('<param name="eux_15" value="EUXLTL|idc_lite;EUX/LTL;Valuta;0" />');		
		document.writeln('<param name="eux_16" value="EUXMXN|idc_lite;EUX/MXN;Valuta;0" />');
		document.writeln('<param name="eux_17" value="EUXMYR|idc_lite;EUX/MYR;Valuta;0" />');
		document.writeln('<param name="eux_18" value="EUXNOK|idc_lite;EUX/NOK;Valuta;0" />');
		document.writeln('<param name="eux_19" value="EUXNZD|idc_lite;EUX/NZD;Valuta;0" />');
		document.writeln('<param name="eux_20" value="EUXPHP|idc_lite;EUX/PHP;Valuta;0" />');
		document.writeln('<param name="eux_21" value="EUXPLN|idc_lite;EUX/PLN;Valuta;0" />');
		document.writeln('<param name="eux_22" value="EUXRON|idc_lite;EUX/RON;Valuta;0" />');
		document.writeln('<param name="eux_23" value="EUXSEK|idc_lite;EUX/SEK;Valuta;0" />');
		document.writeln('<param name="eux_24" value="USDHKD|idc_lite;USD/HKD;Valuta;0" />');
		document.writeln('<param name="eux_25" value="EUXTHB|idc_lite;EUX/THB;Valuta;0" />');
		document.writeln('<param name="eux_26" value="EUXUSD|idc_lite;EUX/USD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_150_path" value="Cross Rates" />');	
		
		document.writeln('<param name="iexplore_market_151" value="@HTML|ern_|Eritrean Nakfa|false|true|false|false" />');
		document.writeln('<param name="ern_1" value="ERNUSD|idc_lite;ERN/USD;Valuta;0" />');
		document.writeln('<param name="ern_2" value="AUDERN|idc_lite;AUD/ERN;Valuta;0" />');
		document.writeln('<param name="ern_3" value="CADERN|idc_lite;CAD/ERN;Valuta;0" />');
		document.writeln('<param name="ern_4" value="CHFERN|idc_lite;CHF/ERN;Valuta;0" />');
		document.writeln('<param name="ern_5" value="EURERN|idc_lite;EUR/ERN;Valuta;0" />');
		document.writeln('<param name="ern_6" value="GBPERN|idc_lite;GBP/ERN;Valuta;0" />');
		document.writeln('<param name="ern_7" value="JPYERN|idc_lite;JPY/ERN;Valuta;0" />');
		document.writeln('<param name="ern_8" value="USDERN|idc_lite;USD/ERN;Valuta;0" />');
		document.writeln('<param name="iexplore_market_151_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_152" value="@HTML|gmd_|Gambian Dalasi|false|true|false|false" />');
		document.writeln('<param name="gmd_1" value="GMDGBP|idc_lite;GMD/GBP;Valuta;0" />');
		document.writeln('<param name="gmd_2" value="CADGMD|idc_lite;CAD/GMD;Valuta;0" />');
		document.writeln('<param name="gmd_3" value="CHFGMD|idc_lite;CHF/GMD;Valuta;0" />');
		document.writeln('<param name="gmd_4" value="EURGMD|idc_lite;EUR/GMD;Valuta;0" />');
		document.writeln('<param name="gmd_5" value="GBPGMD|idc_lite;GBP/GMD;Valuta;0" />');	
		document.writeln('<param name="gmd_6" value="JPYGMD|idc_lite;JPY/GMD;Valuta;0" />');
		document.writeln('<param name="gmd_7" value="USDGMD|idc_lite;USD/GMD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_152_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_153" value="@HTML|ghs_|Ghanaian Cedi|false|true|false|false" />');
		document.writeln('<param name="ghs_1" value="GHSEUR|idc_lite;GHS/EUR;Valuta;0" />');
		document.writeln('<param name="ghs_2" value="CADGHS|idc_lite;CAD/GHS;Valuta;0" />');
		document.writeln('<param name="ghs_3" value="CHFGHS|idc_lite;CHF/GHS;Valuta;0" />');
		document.writeln('<param name="ghs_4" value="EURGHS|idc_lite;EUR/GHS;Valuta;0" />');
		document.writeln('<param name="ghs_5" value="GBPGHS|idc_lite;GBP/GHS;Valuta;0" />');
		document.writeln('<param name="ghs_6" value="JPYGHS|idc_lite;JPY/GHS;Valuta;0" />');
		document.writeln('<param name="ghs_7" value="SGDGHS|idc_lite;SGD/GHS;Valuta;0" />');
		document.writeln('<param name="ghs_8" value="USDGHS|idc_lite;USD/GHS;Valuta;0" />');
		document.writeln('<param name="iexplore_market_153_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_154" value="@HTML|gip_|Gibraltar Pound|false|true|false|false" />');
		document.writeln('<param name="gip_1" value="GIPUSD|idc_lite;GIP/USD;Valuta;0" />');
		document.writeln('<param name="gip_2" value="AUDGIP|idc_lite;AUD/GIP;Valuta;0" />');
		document.writeln('<param name="gip_3" value="CADGIP|idc_lite;CAD/GIP;Valuta;0" />');
		document.writeln('<param name="gip_4" value="CHFGIP|idc_lite;CHF/GIP;Valuta;0" />');
		document.writeln('<param name="gip_5" value="DKKGIP|idc_lite;DKK/GIP;Valuta;0" />');
		document.writeln('<param name="gip_6" value="EURGIP|idc_lite;EUR/GIP;Valuta;0" />');
		document.writeln('<param name="gip_7" value="GBPGIP|idc_lite;GBP/GIP;Valuta;0" />');	
		document.writeln('<param name="gip_8" value="JPYGIP|idc_lite;JPY/GIP;Valuta;0" />');
		document.writeln('<param name="gip_9" value="SEKGIP|idc_lite;SEK/GIP;Valuta;0" />');
		document.writeln('<param name="gip_10" value="USDGIP|idc_lite;USD/GIP;Valuta;0" />');
		document.writeln('<param name="iexplore_market_154_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_155" value="@HTML|gyd_|Guyanese Dollar|false|true|false|false" />');
		document.writeln('<param name="gyd_1" value="GYDUSD|idc_lite;GYD/USD;Valuta;0" />');
		document.writeln('<param name="gyd_2" value="CADGYD|idc_lite;CAD/GYD;Valuta;0" />');
		document.writeln('<param name="gyd_3" value="CHFGYD|idc_lite;CHF/GYD;Valuta;0" />');
		document.writeln('<param name="gyd_4" value="EURGYD|idc_lite;EUR/GYD;Valuta;0" />');
		document.writeln('<param name="gyd_5" value="GBPGYD|idc_lite;GBP/GYD;Valuta;0" />');
		document.writeln('<param name="gyd_6" value="JPYGYD|idc_lite;JPY/GYD;Valuta;0" />');
		document.writeln('<param name="gyd_7" value="USDGYD|idc_lite;USD/GYD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_155_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_156" value="@HTML|hkd_|Hong Kong Dollar|false|true|false|false" />');
		document.writeln('<param name="hkd_1" value="EURHKD|idc_lite;EUR/HKD;Valuta;0" />');
		document.writeln('<param name="hkd_2" value="GBPHKD|idc_lite;GBP/HKD;Valuta;0" />');
		document.writeln('<param name="hkd_3" value="USDHKD|idc_lite;USD/HKD;Valuta;0" />');
		document.writeln('<param name="hkd_4" value="HKDAUD|idc_lite;HKD/AUD;Valuta;0" />');
		document.writeln('<param name="hkd_5" value="HKDBRL|idc_lite;HKD/BRL;Valuta;0" />');
		document.writeln('<param name="hkd_6" value="HKDCAD|idc_lite;HKD/CAD;Valuta;0" />');
		document.writeln('<param name="hkd_7" value="HKDCHF|idc_lite;HKD/CHF;Valuta;0" />');
		document.writeln('<param name="hkd_8" value="HKDCNY|idc_lite;HKD/CNY;Valuta;0" />');
		document.writeln('<param name="hkd_9" value="HKDDKK|idc_lite;HKD/DKK;Valuta;0" />');
		document.writeln('<param name="hkd_10" value="HKDEUR|idc_lite;HKD/EUR;Valuta;0" />');
		document.writeln('<param name="hkd_11" value="HKDGBP|idc_lite;HKD/GBP;Valuta;0" />');
		document.writeln('<param name="hkd_12" value="HKDIDR|idc_lite;HKD/IDR;Valuta;0" />');
		document.writeln('<param name="hkd_13" value="HKDINR|idc_lite;HKD/INR;Valuta;0" />');
		document.writeln('<param name="hkd_14" value="HKDJPY|idc_lite;HKD/JPY;Valuta;0" />');
		document.writeln('<param name="hkd_15" value="HKDKRW|idc_lite;HKD/KRW;Valuta;0" />');
		document.writeln('<param name="hkd_16" value="HKDMXN|idc_lite;HKD/MXN;Valuta;0" />');
		document.writeln('<param name="hkd_17" value="HKDMYR|idc_lite;HKD/MYR;Valuta;0" />');
		document.writeln('<param name="hkd_18" value="HKDNZD|idc_lite;HKD/NZD;Valuta;0" />');
		document.writeln('<param name="hkd_19" value="HKDPHP|idc_lite;HKD/PHP;Valuta;0" />');		
		document.writeln('<param name="hkd_20" value="HKDPKR|idc_lite;HKD/PKR;Valuta;0" />');
		document.writeln('<param name="hkd_21" value="HKDPLN|idc_lite;HKD/PLN;Valuta;0" />');
		document.writeln('<param name="hkd_22" value="HKDSEK|idc_lite;HKD/SEK;Valuta;0" />');
		document.writeln('<param name="hkd_23" value="HKDSGD|idc_lite;HKD/SGD;Valuta;0" />');
		document.writeln('<param name="hkd_24" value="HKDTHB|idc_lite;HKD/THB;Valuta;0" />');
		document.writeln('<param name="hkd_25" value="HKDTWD|idc_lite;HKD/TWD;Valuta;0" />');
		document.writeln('<param name="hkd_26" value="HKDUSD|idc_lite;HKD/USD;Valuta;0" />');
		document.writeln('<param name="hkd_27" value="HKDVND|idc_lite;HKD/VND;Valuta;0" />');
		document.writeln('<param name="hkd_28" value="HKDZAR|idc_lite;HKD/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_156_path" value="Cross Rates" />');


		document.writeln('<param name="iexplore_market_157" value="@HTML|huf_|Hungarian Forint|false|true|false|false" />');
		document.writeln('<param name="huf_1" value="CHFHUF|idc_lite;CHF/HUF;Valuta;0" />');
		document.writeln('<param name="huf_2" value="EURHUF|idc_lite;EUR/HUF;Valuta;0" />');
		document.writeln('<param name="huf_3" value="GBPHUF|idc_lite;GBP/HUF;Valuta;0" />');
		document.writeln('<param name="huf_4" value="USDHUF|idc_lite;USD/HUF;Valuta;0" />');
		document.writeln('<param name="huf_5" value="HUFCHF|idc_lite;HUF/CHF;Valuta;0" />');		
		document.writeln('<param name="huf_6" value="HUFCZK|idc_lite;HUF/CZK;Valuta;0" />');			
		document.writeln('<param name="huf_7" value="HUFDKK|idc_lite;HUF/DKK;Valuta;0" />');	
		document.writeln('<param name="huf_8" value="HUFEUR|idc_lite;HUF/EUR;Valuta;0" />');		
		document.writeln('<param name="huf_9" value="HUFGBP|idc_lite;HUF/GBP;Valuta;0" />');			
		document.writeln('<param name="huf_10" value="HUFJPY|idc_lite;HUF/JPY;Valuta;0" />');	
		document.writeln('<param name="huf_11" value="HUFRON|idc_lite;HUF/RON;Valuta;0" />');		
		document.writeln('<param name="huf_12" value="HUFSEK|idc_lite;HUF/SEK;Valuta;0" />');			
		document.writeln('<param name="huf_13" value="HUFUSD|idc_lite;HUF/USD;Valuta;0" />');			
		document.writeln('<param name="iexplore_market_157_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_158" value="@HTML|isk_|Icelandic Krona|false|true|false|false" />');
		document.writeln('<param name="isk_1" value="ISKEUR|idc_lite;ISK/EUR;Valuta;0" />');
		document.writeln('<param name="isk_2" value="ISKGBP|idc_lite;ISK/GBP;Valuta;0" />');
		document.writeln('<param name="isk_3" value="USDISK|idc_lite;USD/ISK;Valuta;0" />');
		document.writeln('<param name="isk_4" value="ISKCAD|idc_lite;ISK/CAD;Valuta;0" />');
		document.writeln('<param name="isk_5" value="ISKCHF|idc_lite;ISK/CHF;Valuta;0" />');
		document.writeln('<param name="isk_6" value="ISKDKK|idc_lite;ISK/DKK;Valuta;0" />');
		document.writeln('<param name="isk_7" value="ISKSEK|idc_lite;ISK/SEK;Valuta;0" />');
		document.writeln('<param name="iexplore_market_158_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_159" value="@HTML|inr_|Indian Rupee|false|true|false|false" />');
		document.writeln('<param name="inr_1" value="EURINR|idc_lite;EUR/INR;Valuta;0" />');
		document.writeln('<param name="inr_2" value="GBPINR|idc_lite;GBP/INR;Valuta;0" />');
		document.writeln('<param name="inr_3" value="INRJPY|idc_lite;INR/JPY;Valuta;0" />');
		document.writeln('<param name="inr_4" value="USDINR|idc_lite;USD/INR;Valuta;0" />');	
		document.writeln('<param name="inr_5" value="INRAUD|idc_lite;INR/AUD;Valuta;0" />');
		document.writeln('<param name="inr_6" value="INRCAD|idc_lite;INR/CAD;Valuta;0" />');
		document.writeln('<param name="inr_7" value="INRCHF|idc_lite;INR/CHF;Valuta;0" />');
		document.writeln('<param name="inr_8" value="INRCNY|idc_lite;INR/CNY;Valuta;0" />');
		document.writeln('<param name="inr_9" value="INREUR|idc_lite;INR/EUR;Valuta;0" />');
		document.writeln('<param name="inr_10" value="INRGBP|idc_lite;INR/GBP;Valuta;0" />');
		document.writeln('<param name="inr_11" value="INRHKD|idc_lite;INR/HKD;Valuta;0" />');
		document.writeln('<param name="inr_124 value="INRIDR|idc_lite;INR/IDR;Valuta;0" />');	
		document.writeln('<param name="inr_13" value="INRJPY|idc_lite;INR/JPY;Valuta;0" />');
		document.writeln('<param name="inr_14" value="INRKRW|idc_lite;INR/KRW;Valuta;0" />');
		document.writeln('<param name="inr_15" value="INRMYR|idc_lite;INR/MYR;Valuta;0" />');
		document.writeln('<param name="inr_16" value="INRNZD|idc_lite;INR/NZD;Valuta;0" />');
		document.writeln('<param name="inr_17" value="INRPHP|idc_lite;INR/PHP;Valuta;0" />');
		document.writeln('<param name="inr_18" value="INRPKR|idc_lite;INR/PKR;Valuta;0" />');
		document.writeln('<param name="inr_19" value="INRSGD|idc_lite;INR/SGD;Valuta;0" />');
		document.writeln('<param name="inr_20" value="INRTHB|idc_lite;INR/THB;Valuta;0" />');
		document.writeln('<param name="inr_21" value="INRTWD|idc_lite;INR/TWD;Valuta;0" />');
		document.writeln('<param name="inr_22" value="INRUSD|idc_lite;INR/USD;Valuta;0" />');
		document.writeln('<param name="inr_23" value="INRXCU|idc_lite;INR/XCU;Valuta;0" />');
		document.writeln('<param name="inr_24" value="INRZAR|idc_lite;INR/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_159_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_160" value="@HTML|idr_|Indonesian Rupiah|false|true|false|false" />');
		document.writeln('<param name="idr_1" value="EURIDR|idc_lite;EUR/IDR;Valuta;0" />');
		document.writeln('<param name="idr_2" value="GBPIDR|idc_lite;GBP/IDR;Valuta;0" />');
		document.writeln('<param name="idr_3" value="IDRJPY|idc_lite;IDR/JPY;Valuta;0" />');
		document.writeln('<param name="idr_4" value="USDIDR|idc_lite;USD/IDR;Valuta;0" />');
		document.writeln('<param name="idr_5" value="IDRAUD|idc_lite;IDR/AUD;Valuta;0" />');
		document.writeln('<param name="idr_6" value="IDRBRL|idc_lite;IDR/BRL;Valuta;0" />');
		document.writeln('<param name="idr_7" value="IDRCAD|idc_lite;IDR/CAD;Valuta;0" />');
		document.writeln('<param name="idr_8" value="IDRCHF|idc_lite;IDR/CHF;Valuta;0" />');		
		document.writeln('<param name="idr_9" value="IDRCNY|idc_lite;IDR/CNY;Valuta;0" />');
		document.writeln('<param name="idr_10" value="IDREUR|idc_lite;IDR/EUR;Valuta;0" />');
		document.writeln('<param name="idr_11" value="IDRGBP|idc_lite;IDR/GBP;Valuta;0" />');
		document.writeln('<param name="idr_12" value="IDRHKD|idc_lite;IDR/HKD;Valuta;0" />');			
		document.writeln('<param name="idr_13" value="IDRINR|idc_lite;IDR/INR;Valuta;0" />');
		document.writeln('<param name="idr_14" value="IDRJPY|idc_lite;IDR/JPY;Valuta;0" />');
		document.writeln('<param name="idr_15" value="IDRKRW|idc_lite;IDR/KRW;Valuta;0" />');
		document.writeln('<param name="idr_16" value="IDRMYR|idc_lite;IDR/MYR;Valuta;0" />');	
		document.writeln('<param name="idr_17" value="IDRNZD|idc_lite;IDR/NZD;Valuta;0" />');
		document.writeln('<param name="idr_18" value="IDRPHP|idc_lite;IDR/PHP;Valuta;0" />');
		document.writeln('<param name="idr_19" value="IDRSGD|idc_lite;IDR/SGD;Valuta;0" />');
		document.writeln('<param name="idr_20" value="IDRTHB|idc_lite;IDR/THB;Valuta;0" />');		
		document.writeln('<param name="idr_21" value="IDRTWD|idc_lite;IDR/TWD;Valuta;0" />');		
		document.writeln('<param name="idr_22" value="IDRUSD|idc_lite;IDR/USD;Valuta;0" />');		
		document.writeln('<param name="idr_23" value="IDRXCU|idc_lite;IDR/XCU;Valuta;0" />');		
		document.writeln('<param name="idr_24" value="IDRZAR|idc_lite;IDR/ZAR;Valuta;0" />');				
		document.writeln('<param name="iexplore_market_160_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_161" value="@HTML|irr_|Iran Riyal|false|true|false|false" />');
		document.writeln('<param name="irr_1" value="EURIRR|idc_lite;EUR/IRR;Valuta;0" />');
		document.writeln('<param name="irr_2" value="USDIRR|idc_lite;USD/IRR;Valuta;0" />');
		document.writeln('<param name="irr_3" value="CADIRR|idc_lite;CAD/IRR;Valuta;0" />');
		document.writeln('<param name="irr_4" value="CHFIRR|idc_lite;CHF/IRR;Valuta;0" />');
		document.writeln('<param name="irr_5" value="GBPIRR|idc_lite;GBP/IRR;Valuta;0" />');
		document.writeln('<param name="irr_6" value="JPYIRR|idc_lite;JPY/IRR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_161_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_162" value="@HTML|iqd_|Iraqi Dinar|false|true|false|false" />');
		document.writeln('<param name="iqd_1" value="EURIQD|idc_lite;EUR/IQD;Valuta;0" />');
		document.writeln('<param name="iqd_2" value="USDIQD|idc_lite;USD/IQD;Valuta;0" />');
		document.writeln('<param name="iqd_3" value="GBPIQD|idc_lite;GBP/IQD;Valuta;0" />');
		document.writeln('<param name="iqd_4" value="IQDUSD|idc_lite;IQD/USD;Valuta;0" />');
		document.writeln('<param name="iqd_5" value="CHFIQD|idc_lite;CHF/IQD;Valuta;0" />');
		document.writeln('<param name="iqd_6" value="CADIQD|idc_lite;CAD/IQD;Valuta;0" />');
		document.writeln('<param name="iqd_6" value="JPYIQD|idc_lite;JPY/IQD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_162_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_163" value="@HTML|ils_|Israeli Shekel|false|true|false|false" />');
		document.writeln('<param name="ils_1" value="AUDILS|idc_lite;AUD/ILS;Valuta;0" />');
		document.writeln('<param name="ils_2" value="CADILS|idc_lite;CAD/ILS;Valuta;0" />');
		document.writeln('<param name="ils_3" value="CHFILS|idc_lite;CHF/ILS;Valuta;0" />');
		document.writeln('<param name="ils_4" value="EURILS|idc_lite;EUR/ILS;Valuta;0" />');
		document.writeln('<param name="ils_5" value="GBPILS|idc_lite;GBP/ILS;Valuta;0" />');
		document.writeln('<param name="ils_6" value="JPYILS|idc_lite;JPY/ILS;Valuta;0" />');
		document.writeln('<param name="ils_7" value="USDILS|idc_lite;USD/ILS;Valuta;0" />');
		document.writeln('<param name="ils_8" value="ILSBRL|idc_lite;ILS/BRL;Valuta;0" />');
		document.writeln('<param name="ils_9" value="ILSCHF|idc_lite;ILS/CHF;Valuta;0" />');
		document.writeln('<param name="ils_10" value="ILSEUR|idc_lite;ILS/EUR;Valuta;0" />');
		document.writeln('<param name="ils_11" value="ILSGBP|idc_lite;ILS/GBP;Valuta;0" />');
		document.writeln('<param name="ils_12" value="ILSJPY|idc_lite;ILS/JPY;Valuta;0" />');
		document.writeln('<param name="ils_13" value="ILSUSD|idc_lite;ILS/USD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_163_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_164" value="@HTML|jpy_|Japanese Yen|false|true|false|false" />');
		document.writeln('<param name="jpy_1" value="AUDJPY|idc_lite;AUD/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_2" value="CADJPY|idc_lite;CAD/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_3" value="CHFJPY|idc_lite;CHF/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_4" value="EURJPY|idc_lite;EUR/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_5" value="GBPJPY|idc_lite;GBP/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_6" value="HKDJPY|idc_lite;HKD/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_7" value="IDRJPY|idc_lite;IDR/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_8" value="INRJPY|idc_lite;INR/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_9" value="KRWJPY|idc_lite;KRW/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_10" value="NZDJPY|idc_lite;NZD/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_11" value="SGDJPY|idc_lite;SGD/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_12" value="THBJPY|idc_lite;THB/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_13" value="TWDJPY|idc_lite;TWD/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_14" value="USDJPY|idc_lite;USD/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_15" value="ZARJPY|idc_lite;ZAR/JPY;Valuta;0" />');
		document.writeln('<param name="jpy_16" value="JPYAED|idc_lite;JPY/AED;Valuta;0" />');
		document.writeln('<param name="jpy_17" value="JPYAFN|idc_lite;JPY/AFN;Valuta;0" />');
		document.writeln('<param name="jpy_18" value="JPYALL|idc_lite;JPY/ALL;Valuta;0" />');
		document.writeln('<param name="jpy_19" value="JPYANG|idc_lite;JPY/ANG;Valuta;0" />');
		document.writeln('<param name="jpy_20" value="JPYARS|idc_lite;JPY/ARS;Valuta;0" />');
		document.writeln('<param name="jpy_21" value="JPYAUD|idc_lite;JPY/AUD;Valuta;0" />');
		document.writeln('<param name="jpy_22" value="JPYAWG|idc_lite;JPY/AWG;Valuta;0" />');
		document.writeln('<param name="jpy_23" value="JPYBBD|idc_lite;JPY/BBD;Valuta;0" />');
		document.writeln('<param name="jpy_24" value="JPYBDT|idc_lite;JPY/BDT;Valuta;0" />');
		document.writeln('<param name="jpy_25" value="JPYBGN|idc_lite;JPY/BGN;Valuta;0" />');
		document.writeln('<param name="jpy_26" value="JPYBHD|idc_lite;JPY/BHD;Valuta;0" />');
		document.writeln('<param name="jpy_27" value="JPYBIF|idc_lite;JPY/BIF;Valuta;0" />');
		document.writeln('<param name="jpy_28" value="JPYBMD|idc_lite;JPY/BMD;Valuta;0" />');
		document.writeln('<param name="jpy_29" value="JPYBND|idc_lite;JPY/BND;Valuta;0" />');
		document.writeln('<param name="jpy_30" value="JPYBOB|idc_lite;JPY/BOB;Valuta;0" />');
		document.writeln('<param name="jpy_31" value="JPYBRL|idc_lite;JPY/BRL;Valuta;0" />');
		document.writeln('<param name="jpy_32" value="JPYBSD|idc_lite;JPY/BSD;Valuta;0" />');
		document.writeln('<param name="jpy_33" value="JPYBTN|idc_lite;JPY/BTN;Valuta;0" />');
		document.writeln('<param name="jpy_34" value="JPYBWP|idc_lite;JPY/BWP;Valuta;0" />');
		document.writeln('<param name="jpy_35" value="JPYBYR|idc_lite;JPY/BYR;Valuta;0" />');
		document.writeln('<param name="jpy_36" value="JPYBZD|idc_lite;JPY/BZD;Valuta;0" />');
		document.writeln('<param name="jpy_37" value="JPYCAD|idc_lite;JPY/CAD;Valuta;0" />');
		document.writeln('<param name="jpy_38" value="JPYCDF|idc_lite;JPY/CDF;Valuta;0" />');
		document.writeln('<param name="jpy_39" value="JPYCHF|idc_lite;JPY/CHF;Valuta;0" />');
		document.writeln('<param name="jpy_40" value="JPYCLP|idc_lite;JPY/CLP;Valuta;0" />');
		document.writeln('<param name="jpy_41" value="JPYCNY|idc_lite;JPY/CNY;Valuta;0" />');
		document.writeln('<param name="jpy_42" value="JPYCOP|idc_lite;JPY/COP;Valuta;0" />');
		document.writeln('<param name="jpy_43" value="JPYCRC|idc_lite;JPY/CRC;Valuta;0" />');
		document.writeln('<param name="jpy_44" value="JPYCUP|idc_lite;JPY/CUP;Valuta;0" />');
		document.writeln('<param name="jpy_45" value="JPYCVE|idc_lite;JPY/CVE;Valuta;0" />');
		document.writeln('<param name="jpy_46" value="JPYCZK|idc_lite;JPY/CZK;Valuta;0" />');
		document.writeln('<param name="jpy_47" value="JPYDJF|idc_lite;JPY/DJF;Valuta;0" />');
		document.writeln('<param name="jpy_48" value="JPYDKK|idc_lite;JPY/DKK;Valuta;0" />');
		document.writeln('<param name="jpy_49" value="JPYDOP|idc_lite;JPY/DOP;Valuta;0" />');
		document.writeln('<param name="jpy_50" value="JPYDZD|idc_lite;JPY/DZD;Valuta;0" />');
		document.writeln('<param name="jpy_51" value="JPYEGP|idc_lite;JPY/EGP;Valuta;0" />');
		document.writeln('<param name="jpy_52" value="JPYERN|idc_lite;JPY/ERN;Valuta;0" />');
		document.writeln('<param name="jpy_53" value="JPYEUR|idc_lite;JPY/EUR;Valuta;0" />');
		document.writeln('<param name="jpy_54" value="JPYFKP|idc_lite;JPY/FKP;Valuta;0" />');
		document.writeln('<param name="jpy_55" value="JPYGBP|idc_lite;JPY/GBP;Valuta;0" />');
		document.writeln('<param name="jpy_56" value="JPYGEL|idc_lite;JPY/GEL;Valuta;0" />');
		document.writeln('<param name="jpy_57" value="JPYGHS|idc_lite;JPY/GHS;Valuta;0" />');
		document.writeln('<param name="jpy_58" value="JPYGIP|idc_lite;JPY/GIP;Valuta;0" />');
		document.writeln('<param name="jpy_59" value="JPYGMD|idc_lite;JPY/GMD;Valuta;0" />');
		document.writeln('<param name="jpy_60" value="JPYGNF|idc_lite;JPY/GNF;Valuta;0" />');
		document.writeln('<param name="jpy_61" value="JPYGTQ|idc_lite;JPY/GTQ;Valuta;0" />');
		document.writeln('<param name="jpy_62" value="JPYGYD|idc_lite;JPY/GYD;Valuta;0" />');
		document.writeln('<param name="jpy_63" value="JPYHKD|idc_lite;JPY/HKD;Valuta;0" />');
		document.writeln('<param name="jpy_64" value="JPYHNL|idc_lite;JPY/HNL;Valuta;0" />');
		document.writeln('<param name="jpy_65" value="JPYHRK|idc_lite;JPY/HRK;Valuta;0" />');
		document.writeln('<param name="jpy_66" value="JPYHTG|idc_lite;JPY/HTG;Valuta;0" />');
		document.writeln('<param name="jpy_67" value="JPYHUF|idc_lite;JPY/HUF;Valuta;0" />');
		document.writeln('<param name="jpy_68" value="JPYIDR|idc_lite;JPY/IDR;Valuta;0" />');
		document.writeln('<param name="jpy_69" value="JPYILS|idc_lite;JPY/ILS;Valuta;0" />');
		document.writeln('<param name="jpy_70" value="JPYINR|idc_lite;JPY/INR;Valuta;0" />');
		document.writeln('<param name="jpy_71" value="JPYIQD|idc_lite;JPY/IQD;Valuta;0" />');
		document.writeln('<param name="jpy_72" value="JPYIRR|idc_lite;JPY/IRR;Valuta;0" />');
		document.writeln('<param name="jpy_73" value="JPYJMD|idc_lite;JPY/JMD;Valuta;0" />');
		document.writeln('<param name="jpy_74" value="JPYJOD|idc_lite;JPY/JOD;Valuta;0" />');
		document.writeln('<param name="jpy_75" value="JPYKES|idc_lite;JPY/KES;Valuta;0" />');
		document.writeln('<param name="jpy_76" value="JPYKGS|idc_lite;JPY/KGS;Valuta;0" />');
		document.writeln('<param name="jpy_77" value="JPYKHR|idc_lite;JPY/KHR;Valuta;0" />');
		document.writeln('<param name="jpy_78" value="JPYKMF|idc_lite;JPY/KMF;Valuta;0" />');
		document.writeln('<param name="jpy_79" value="JPYKPW|idc_lite;JPY/KPW;Valuta;0" />');
		document.writeln('<param name="jpy_80" value="JPYKRW|idc_lite;JPY/KRW;Valuta;0" />');
		document.writeln('<param name="jpy_81" value="JPYKWD|idc_lite;JPY/KWD;Valuta;0" />');
		document.writeln('<param name="jpy_82" value="JPYKYD|idc_lite;JPY/KYD;Valuta;0" />');
		document.writeln('<param name="jpy_83" value="JPYKZT|idc_lite;JPY/KZT;Valuta;0" />');
		document.writeln('<param name="jpy_84" value="JPYLAK|idc_lite;JPY/LAK;Valuta;0" />');
		document.writeln('<param name="jpy_85" value="JPYLBP|idc_lite;JPY/LBP;Valuta;0" />');
		document.writeln('<param name="jpy_86" value="JPYLKR|idc_lite;JPY/LKR;Valuta;0" />');
		document.writeln('<param name="jpy_87" value="JPYLRD|idc_lite;JPY/LRD;Valuta;0" />');
		document.writeln('<param name="jpy_88" value="JPYLSL|idc_lite;JPY/LSL;Valuta;0" />');
		document.writeln('<param name="jpy_89" value="JPYLTL|idc_lite;JPY/LTL;Valuta;0" />');		
		document.writeln('<param name="jpy_90" value="JPYLYD|idc_lite;JPY/LYD;Valuta;0" />');
		document.writeln('<param name="jpy_91" value="JPYMAD|idc_lite;JPY/MAD;Valuta;0" />');
		document.writeln('<param name="jpy_92" value="JPYMDL|idc_lite;JPY/MDL;Valuta;0" />');
		document.writeln('<param name="jpy_93" value="JPYMGA|idc_lite;JPY/MGA;Valuta;0" />');
		document.writeln('<param name="jpy_94" value="JPYMKD|idc_lite;JPY/MKD;Valuta;0" />');
		document.writeln('<param name="jpy_95" value="JPYMMK|idc_lite;JPY/MMK;Valuta;0" />');
		document.writeln('<param name="jpy_96" value="JPYMNT|idc_lite;JPY/MNT;Valuta;0" />');
		document.writeln('<param name="jpy_97" value="JPYMRO|idc_lite;JPY/MRO;Valuta;0" />');
		document.writeln('<param name="jpy_98" value="JPYMUR|idc_lite;JPY/MUR;Valuta;0" />');
		document.writeln('<param name="jpy_99" value="JPYMVR|idc_lite;JPY/MVR;Valuta;0" />');
		document.writeln('<param name="jpy_100" value="JPYMWK|idc_lite;JPY/MWK;Valuta;0" />');
		document.writeln('<param name="jpy_101" value="JPYMXN|idc_lite;JPY/MXN;Valuta;0" />');
		document.writeln('<param name="jpy_102" value="JPYMYR|idc_lite;JPY/MYR;Valuta;0" />');
		document.writeln('<param name="jpy_103" value="JPYMZN|idc_lite;JPY/MZN;Valuta;0" />');
		document.writeln('<param name="jpy_104" value="JPYNAD|idc_lite;JPY/NAD;Valuta;0" />');
		document.writeln('<param name="jpy_105" value="JPYNGN|idc_lite;JPY/NGN;Valuta;0" />');
		document.writeln('<param name="jpy_106" value="JPYNIO|idc_lite;JPY/NIO;Valuta;0" />');
		document.writeln('<param name="jpy_107" value="JPYNOK|idc_lite;JPY/NOK;Valuta;0" />');
		document.writeln('<param name="jpy_108" value="JPYNPR|idc_lite;JPY/NPR;Valuta;0" />');
		document.writeln('<param name="jpy_109" value="JPYNZD|idc_lite;JPY/NZD;Valuta;0" />');
		document.writeln('<param name="jpy_110" value="JPYOMR|idc_lite;JPY/OMR;Valuta;0" />');
		document.writeln('<param name="jpy_111" value="JPYPAB|idc_lite;JPY/PAB;Valuta;0" />');
		document.writeln('<param name="jpy_112" value="JPYPEN|idc_lite;JPY/PEN;Valuta;0" />');
		document.writeln('<param name="jpy_113" value="JPYPHP|idc_lite;JPY/PHP;Valuta;0" />');
		document.writeln('<param name="jpy_114" value="JPYPKR|idc_lite;JPY/PKR;Valuta;0" />');
		document.writeln('<param name="jpy_115" value="JPYPLN|idc_lite;JPY/PLN;Valuta;0" />');
		document.writeln('<param name="jpy_116" value="JPYPYG|idc_lite;JPY/PYG;Valuta;0" />');
		document.writeln('<param name="jpy_117" value="JPYQAR|idc_lite;JPY/QAR;Valuta;0" />');
		document.writeln('<param name="jpy_118" value="JPYRON|idc_lite;JPY/RON;Valuta;0" />');
		document.writeln('<param name="jpy_119" value="JPYRSD|idc_lite;JPY/RSD;Valuta;0" />');
		document.writeln('<param name="jpy_120" value="JPYRUB|idc_lite;JPY/RUB;Valuta;0" />');
		document.writeln('<param name="jpy_121" value="JPYRWF|idc_lite;JPY/RWF;Valuta;0" />');
		document.writeln('<param name="jpy_122" value="JPYSAR|idc_lite;JPY/SAR;Valuta;0" />');
		document.writeln('<param name="jpy_123" value="JPYSCR|idc_lite;JPY/SCR;Valuta;0" />');
		document.writeln('<param name="jpy_124" value="JPYSDG|idc_lite;JPY/SDG;Valuta;0" />');
		document.writeln('<param name="jpy_125" value="JPYSEK|idc_lite;JPY/SEK;Valuta;0" />');
		document.writeln('<param name="jpy_126" value="JPYSGD|idc_lite;JPY/SGD;Valuta;0" />');
		document.writeln('<param name="jpy_127" value="JPYSHP|idc_lite;JPY/SHP;Valuta;0" />');
		document.writeln('<param name="jpy_128" value="JPYSLL|idc_lite;JPY/SLL;Valuta;0" />');
		document.writeln('<param name="jpy_129" value="JPYSOS|idc_lite;JPY/SOS;Valuta;0" />');
		document.writeln('<param name="jpy_130" value="JPYSRD|idc_lite;JPY/SRD;Valuta;0" />');
		document.writeln('<param name="jpy_131" value="JPYSTD|idc_lite;JPY/STD;Valuta;0" />');
		document.writeln('<param name="jpy_132" value="JPYSVC|idc_lite;JPY/SVC;Valuta;0" />');
		document.writeln('<param name="jpy_133" value="JPYSYP|idc_lite;JPY/SYP;Valuta;0" />');
		document.writeln('<param name="jpy_134" value="JPYSZL|idc_lite;JPY/SZL;Valuta;0" />');
		document.writeln('<param name="jpy_135" value="JPYTHB|idc_lite;JPY/THB;Valuta;0" />');
		document.writeln('<param name="jpy_136" value="JPYTJS|idc_lite;JPY/TJS;Valuta;0" />');
		document.writeln('<param name="jpy_137" value="JPYTMT|idc_lite;JPY/TMT;Valuta;0" />');
		document.writeln('<param name="jpy_138" value="JPYTND|idc_lite;JPY/TND;Valuta;0" />');
		document.writeln('<param name="jpy_139" value="JPYTOP|idc_lite;JPY/TOP;Valuta;0" />');
		document.writeln('<param name="jpy_140" value="JPYTRY|idc_lite;JPY/TRY;Valuta;0" />');
		document.writeln('<param name="jpy_141" value="JPYTTD|idc_lite;JPY/TTD;Valuta;0" />');
		document.writeln('<param name="jpy_142" value="JPYTWD|idc_lite;JPY/TWD;Valuta;0" />');
		document.writeln('<param name="jpy_143" value="JPYTZS|idc_lite;JPY/TZS;Valuta;0" />');
		document.writeln('<param name="jpy_144" value="JPYUAH|idc_lite;JPY/UAH;Valuta;0" />');
		document.writeln('<param name="jpy_145" value="JPYUGX|idc_lite;JPY/UGX;Valuta;0" />');
		document.writeln('<param name="jpy_146" value="JPYUSD|idc_lite;JPY/USD;Valuta;0" />');
		document.writeln('<param name="jpy_147" value="JPYUYU|idc_lite;JPY/UYU;Valuta;0" />');
		document.writeln('<param name="jpy_148" value="JPYVEF|idc_lite;JPY/VEF;Valuta;0" />');
		document.writeln('<param name="jpy_149" value="JPYVND|idc_lite;JPY/VND;Valuta;0" />');
		document.writeln('<param name="jpy_150" value="JPYVUV|idc_lite;JPY/VUV;Valuta;0" />');
		document.writeln('<param name="jpy_151" value="JPYWST|idc_lite;JPY/WST;Valuta;0" />');
		document.writeln('<param name="jpy_152" value="JPYXAF|idc_lite;JPY/XAF;Valuta;0" />');
		document.writeln('<param name="jpy_153" value="JPYXCD|idc_lite;JPY/XCD;Valuta;0" />');
		document.writeln('<param name="jpy_154" value="JPYXCU|idc_lite;JPY/XCU;Valuta;0" />');
		document.writeln('<param name="jpy_155" value="JPYXOF|idc_lite;JPY/XOF;Valuta;0" />');
		document.writeln('<param name="jpy_156" value="JPYXPF|idc_lite;JPY/XPF;Valuta;0" />');
		document.writeln('<param name="jpy_157" value="JPYYER|idc_lite;JPY/YER;Valuta;0" />');
		document.writeln('<param name="jpy_158" value="JPYZAR|idc_lite;JPY/KRW;Valuta;0" />');
		document.writeln('<param name="jpy_159" value="JPYZAR|idc_lite;JPY/ZAR;Valuta;0" />');
		document.writeln('<param name="jpy_160" value="JPYZMW|idc_lite;JPY/ZMW;Valuta;0" />');
		document.writeln('<param name="jpy_161" value="JPYZWL|idc_lite;JPY/ZWL;Valuta;0" />');
		document.writeln('<param name="iexplore_market_164_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_165" value="@HTML|jod_|Jordanian Dinar|false|true|false|false" />');
		document.writeln('<param name="jod_1" value="EURJOD|idc_lite;EUR/JOD;Valuta;0" />');
		document.writeln('<param name="jod_2" value="GBPJOD|idc_lite;GBP/JOD;Valuta;0" />');
		document.writeln('<param name="jod_3" value="USDJOD|idc_lite;USD/JOD;Valuta;0" />');
		document.writeln('<param name="jod_4" value="JODEUR|idc_lite;JOD/EUR;Valuta;0" />');
		document.writeln('<param name="jod_5" value="JODGBP|idc_lite;JOD/GBP;Valuta;0" />');
		document.writeln('<param name="iexplore_market_165_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_166" value="@HTML|kzt_|Kazakhstani Tenge|false|true|false|false" />');
		document.writeln('<param name="kzt_1" value="KZTEUR|idc_lite;KZT/EUR;Valuta;0" />');
		document.writeln('<param name="kzt_2" value="KZTGBP|idc_lite;KZT/GBP;Valuta;0" />');
		document.writeln('<param name="kzt_3" value="CADKZT|idc_lite;CAD/KZT;Valuta;0" />');
		document.writeln('<param name="kzt_4" value="CHFKZT|idc_lite;CHF/KZT;Valuta;0" />');
		document.writeln('<param name="kzt_5" value="EURKZT|idc_lite;EUR/KZT;Valuta;0" />');
		document.writeln('<param name="kzt_6" value="GBPKZT|idc_lite;GBP/KZT;Valuta;0" />');
		document.writeln('<param name="kzt_7" value="JPYKZT|idc_lite;JPY/KZT;Valuta;0" />');
		document.writeln('<param name="kzt_8" value="USDKZT|idc_lite;USD/KZT;Valuta;0" />');
		document.writeln('<param name="iexplore_market_166_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_167" value="@HTML|kes_|Kenyan Shilling|false|true|false|false" />');
		document.writeln('<param name="kes_1" value="KESEUR|idc_lite;KES/EUR;Valuta;0" />');
		document.writeln('<param name="kes_2" value="KESGBP|idc_lite;KES/GBP;Valuta;0" />');
		document.writeln('<param name="kes_3" value="KESZAR|idc_lite;KES/ZAR;Valuta;0" />');
		document.writeln('<param name="kes_4" value="AUDKES|idc_lite;AUD/KES;Valuta;0" />');
		document.writeln('<param name="kes_5" value="CADKES|idc_lite;CAD/KES;Valuta;0" />');
		document.writeln('<param name="kes_6" value="CHFKES|idc_lite;CHF/KES;Valuta;0" />');
		document.writeln('<param name="kes_7" value="EURKES|idc_lite;EUR/KES;Valuta;0" />');
		document.writeln('<param name="kes_8" value="GBPKES|idc_lite;GBP/KES;Valuta;0" />');
		document.writeln('<param name="kes_9" value="JPYKES|idc_lite;JPY/KES;Valuta;0" />');
		document.writeln('<param name="kes_10" value="USDKES|idc_lite;USD/KES;Valuta;0" />');
		document.writeln('<param name="kes_11" value="ZARKES|idc_lite;ZAR/KES;Valuta;0" />');
		document.writeln('<param name="iexplore_market_167_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_168" value="@HTML|krw_|Korean Won|false|true|false|false" />');
		document.writeln('<param name="krw_1" value="EURKRW|idc_lite;EUR/KRW;Valuta;0" />');
		document.writeln('<param name="krw_2" value="GBPKRW|idc_lite;GBP/KRW;Valuta;0" />');
		document.writeln('<param name="krw_3" value="JPYKRW|idc_lite;JPY/KRW;Valuta;0" />');
		document.writeln('<param name="krw_4" value="USDKRW|idc_lite;USD/KRW;Valuta;0" />');		
		document.writeln('<param name="krw_5" value="KRWAUD|idc_lite;KRW/AUD;Valuta;0" />');
		document.writeln('<param name="krw_6" value="KRWCAD|idc_lite;KRW/CAD;Valuta;0" />');
		document.writeln('<param name="krw_7" value="KRWCHF|idc_lite;KRW/CHF;Valuta;0" />');
		document.writeln('<param name="krw_8" value="KRWCNY|idc_lite;KRW/CNY;Valuta;0" />');
		document.writeln('<param name="krw_9" value="KRWCRC|idc_lite;KRW/CRC;Valuta;0" />');
		document.writeln('<param name="krw_10" value="KRWEUR|idc_lite;KRW/EUR;Valuta;0" />');
		document.writeln('<param name="krw_11" value="KRWGBP|idc_lite;KRW/GBP;Valuta;0" />');
		document.writeln('<param name="krw_12" value="KRWHKD|idc_lite;KRW/HKD;Valuta;0" />');
		document.writeln('<param name="krw_13" value="KRWIDR|idc_lite;KRW/IDR;Valuta;0" />');
		document.writeln('<param name="krw_14" value="KRWINR|idc_lite;KRW/INR;Valuta;0" />');
		document.writeln('<param name="krw_15" value="KRWJPY|idc_lite;KRW/JPY;Valuta;0" />');
		document.writeln('<param name="krw_16" value="KRWMYR|idc_lite;KRW/MYR;Valuta;0" />');
		document.writeln('<param name="krw_17" value="KRWNZD|idc_lite;KRW/NZD;Valuta;0" />');
		document.writeln('<param name="krw_18" value="KRWPHP|idc_lite;KRW/PHP;Valuta;0" />');
		document.writeln('<param name="krw_19" value="KRWSEK|idc_lite;KRW/SEK;Valuta;0" />');
		document.writeln('<param name="krw_20" value="KRWSGD|idc_lite;KRW/SGD;Valuta;0" />');
		document.writeln('<param name="krw_21" value="KRWTHB|idc_lite;KRW/THB;Valuta;0" />');
		document.writeln('<param name="krw_22" value="KRWTWD|idc_lite;KRW/TWD;Valuta;0" />');	
		document.writeln('<param name="krw_23" value="KRWUSD|idc_lite;KRW/USD;Valuta;0" />');
		document.writeln('<param name="krw_24" value="KRWXCU|idc_lite;KRW/XCU;Valuta;0" />');
		document.writeln('<param name="krw_25" value="KRWZAR|idc_lite;KRW/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_168_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_169" value="@HTML|kwd_|Kuwaiti Dinar|false|true|false|false" />');
		document.writeln('<param name="kwd_1" value="EURKWD|idc_lite;EUR/KWD;Valuta;0" />');
		document.writeln('<param name="kwd_2" value="GBPKWD|idc_lite;GBP/KWD;Valuta;0" />');
		document.writeln('<param name="kwd_3" value="USDKWD|idc_lite;USD/KWD;Valuta;0" />');
		document.writeln('<param name="kwd_4" value="KWDEUR|idc_lite;KWD/EUR;Valuta;0" />');
		document.writeln('<param name="kwd_5" value="KWDGBP|idc_lite;KWD/GBP;Valuta;0" />');
		document.writeln('<param name="kwd_6" value="KWDHKD|idc_lite;KWD/HKD;Valuta;0" />');
		document.writeln('<param name="kwd_7" value="KWDJPY|idc_lite;KWD/JPY;Valuta;0" />');
		document.writeln('<param name="kwd_8" value="KWDPKR|idc_lite;KWD/PKR;Valuta;0" />');
		document.writeln('<param name="kwd_9" value="KWDRSD|idc_lite;KWD/RSD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_169_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_170" value="@HTML|ltl_|Lithaunian Litas|false|true|false|false" />');
		document.writeln('<param name="ltl_1" value="EURLTL|idc_lite;EUR/LTL;Valuta;0" />');
		document.writeln('<param name="ltl_2" value="GBPLTL|idc_lite;GBP/LTL;Valuta;0" />');
		document.writeln('<param name="ltl_3" value="USDLTL|idc_lite;USD/LTL;Valuta;0" />');
		document.writeln('<param name="ltl_4" value="LTLDKK|idc_lite;LTL/DKK;Valuta;0" />');		
		document.writeln('<param name="ltl_5" value="LTLSEK|idc_lite;LTL/SEK;Valuta;0" />');
		document.writeln('<param name="iexplore_market_170_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_171" value="@HTML|lak_|Lao or Laotian Kip|false|true|false|false" />');
		document.writeln('<param name="lak_1" value="LAKUSD|idc_lite;LAK/USD;Valuta;0" />');
		document.writeln('<param name="lak_2" value="CADLAK|idc_lite;CAD/LAK;Valuta;0" />');
		document.writeln('<param name="lak_3" value="EURLAK|idc_lite;EUR/LAK;Valuta;0" />');
		document.writeln('<param name="lak_4" value="GBPLAK|idc_lite;GBP/LAK;Valuta;0" />');
		document.writeln('<param name="lak_5" value="JPYLAK|idc_lite;JPY/LAK;Valuta;0" />');
		document.writeln('<param name="lak_6" value="USDLAK|idc_lite;USD/LAK;Valuta;0" />');
		document.writeln('<param name="iexplore_market_171_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_172" value="@HTML|mop_|Macau Pataca|false|true|false|false" />');
		document.writeln('<param name="mop_1" value="MOPUSD|idc_lite;MOP/USD;Valuta;0" />');
		document.writeln('<param name="mop_2" value="CADMOP|idc_lite;CAD/MOP;Valuta;0" />');
		document.writeln('<param name="mop_3" value="USDMOP|idc_lite;USD/MOP;Valuta;0" />');	
		document.writeln('<param name="iexplore_market_172_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_173" value="@HTML|mga_|Malagasy Ariary|false|true|false|false" />');
		document.writeln('<param name="mga_1" value="MGAUSD|idc_lite;MGA/USD;Valuta;0" />');
		document.writeln('<param name="mga_2" value="AUDMGA|idc_lite;AUD/MGA;Valuta;0" />');
		document.writeln('<param name="mga_3" value="CADMGA|idc_lite;CAD/MGA;Valuta;0" />');
		document.writeln('<param name="mga_4" value="CHFMGA|idc_lite;CHF/MGA;Valuta;0" />');
		document.writeln('<param name="mga_5" value="EURMGA|idc_lite;EUR/MGA;Valuta;0" />');
		document.writeln('<param name="mga_6" value="GBPMGA|idc_lite;GBP/MGA;Valuta;0" />');
		document.writeln('<param name="mga_7" value="JPYMGA|idc_lite;JPY/MGA;Valuta;0" />');
		document.writeln('<param name="mga_8" value="USDMGA|idc_lite;USD/MGA;Valuta;0" />');
		document.writeln('<param name="iexplore_market_173_path" value="Cross Rates" />');		

		document.writeln('<param name="iexplore_market_174" value="@HTML|myr_|Malaysian Ringgit|false|true|false|false" />');
		document.writeln('<param name="myr_1" value="EURMYR|idc_lite;EUR/MYR;Valuta;0" />');
		document.writeln('<param name="myr_2" value="GBPMYR|idc_lite;GBP/MYR;Valuta;0" />');
		document.writeln('<param name="myr_3" value="USDMYR|idc_lite;USD/MYR;Valuta;0" />');
		document.writeln('<param name="myr_4" value="MYRAUD|idc_lite;MYR/AUD;Valuta;0" />');
		document.writeln('<param name="myr_5" value="MYRCAD|idc_lite;MYR/CAD;Valuta;0" />');
		document.writeln('<param name="myr_6" value="MYRCHF|idc_lite;MYR/CHF;Valuta;0" />');
		document.writeln('<param name="myr_7" value="MYRCNH|idc_lite;MYR/CNH;Valuta;0" />');
		document.writeln('<param name="myr_8" value="MYRCNY|idc_lite;MYR/CNY;Valuta;0" />');
		document.writeln('<param name="myr_9" value="MYREUR|idc_lite;MYR/EUR;Valuta;0" />');
		document.writeln('<param name="myr_10" value="MYRGBP|idc_lite;MYR/GBP;Valuta;0" />');
		document.writeln('<param name="myr_11" value="MYRHKD|idc_lite;MYR/HKD;Valuta;0" />');
		document.writeln('<param name="myr_12" value="MYRIDR|idc_lite;MYR/IDR;Valuta;0" />');
		document.writeln('<param name="myr_13" value="MYRINR|idc_lite;MYR/INR;Valuta;0" />');
		document.writeln('<param name="myr_14" value="MYRJPY|idc_lite;MYR/JPY;Valuta;0" />');
		document.writeln('<param name="myr_15" value="MYRKRW|idc_lite;MYR/KRW;Valuta;0" />');
		document.writeln('<param name="myr_16" value="MYRNZD|idc_lite;MYR/NZD;Valuta;0" />');
		document.writeln('<param name="myr_17" value="MYRPHP|idc_lite;MYR/PHP;Valuta;0" />');
		document.writeln('<param name="myr_18" value="MYRPKR|idc_lite;MYR/PKR;Valuta;0" />');
		document.writeln('<param name="myr_19" value="MYRRUB|idc_lite;MYR/RUB;Valuta;0" />');
		document.writeln('<param name="myr_20" value="MYRSEK|idc_lite;MYR/SEK;Valuta;0" />');
		document.writeln('<param name="myr_21" value="MYRSGD|idc_lite;MYR/SGD;Valuta;0" />');
		document.writeln('<param name="myr_22" value="MYRTHB|idc_lite;MYR/THB;Valuta;0" />');
		document.writeln('<param name="myr_23" value="MYRTND|idc_lite;MYR/TND;Valuta;0" />');
		document.writeln('<param name="myr_24" value="MYRTWD|idc_lite;MYR/TWD;Valuta;0" />');
		document.writeln('<param name="myr_25" value="MYRUSD|idc_lite;MYR/USD;Valuta;0" />');
		document.writeln('<param name="myr_26" value="MYRZAR|idc_lite;MYR/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_174_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_175" value="@HTML|mwk_|Malawian Kwacha|false|true|false|false" />');
		document.writeln('<param name="mwk_1" value="MWKGBP|idc_lite;MWK/GBP;Valuta;0" />');
		document.writeln('<param name="mwk_2" value="CADMWK|idc_lite;CAD/MWK;Valuta;0" />');
		document.writeln('<param name="mwk_3" value="CHFMWK|idc_lite;CHF/MWK;Valuta;0" />');
		document.writeln('<param name="mwk_4" value="EURMWK|idc_lite;EUR/MWK;Valuta;0" />');
		document.writeln('<param name="mwk_5" value="GBPMWK|idc_lite;GBP/MWK;Valuta;0" />');
		document.writeln('<param name="mwk_6" value="JPYMWK|idc_lite;JPY/MWK;Valuta;0" />');
		document.writeln('<param name="mwk_7" value="USDMWK|idc_lite;USD/MWK;Valuta;0" />');
		document.writeln('<param name="iexplore_market_175_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_176" value="@HTML|mur_|Mauritian Rupee|false|true|false|false" />');
		document.writeln('<param name="mur_1" value="MURCHF|idc_lite;MUR/CHF;Valuta;0" />');
		document.writeln('<param name="mur_2" value="MUREUR|idc_lite;MUR/EUR;Valuta;0" />');
		document.writeln('<param name="mur_3" value="MURGBP|idc_lite;MUR/GBP;Valuta;0" />');
		document.writeln('<param name="mur_4" value="MURUSD|idc_lite;MUR/USD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_176_path" value="Cross Rates" />');
	
		document.writeln('<param name="iexplore_market_177" value="@HTML|mxn_|Mexican Peso|false|true|false|false" />');
		document.writeln('<param name="mxn_1" value="EURMXN|idc_lite;EUR/MXN;Valuta;0" />');
		document.writeln('<param name="mxn_2" value="GBPMXN|idc_lite;GBP/MXN;Valuta;0" />');
		document.writeln('<param name="mxn_3" value="USDMXN|idc_lite;USD/MXN;Valuta;0" />');
		document.writeln('<param name="mxn_4" value="MXNAUD|idc_lite;MXN/AUD;Valuta;0" />');
		document.writeln('<param name="mxn_5" value="MXNBRL|idc_lite;MXN/BRL;Valuta;0" />');
		document.writeln('<param name="mxn_6" value="MXNCAD|idc_lite;MXN/CAD;Valuta;0" />');
		document.writeln('<param name="mxn_7" value="MXNCHF|idc_lite;MXN/CHF;Valuta;0" />');
		document.writeln('<param name="mxn_8" value="MXNCUC|idc_lite;MXN/CUC;Valuta;0" />');
		document.writeln('<param name="mxn_9" value="MXNDKK|idc_lite;MXN/DKK;Valuta;0" />');		
		document.writeln('<param name="mxn_10" value="MXNEUR|idc_lite;MXN/EUR;Valuta;0" />');
		document.writeln('<param name="mxn_11" value="MXNGBP|idc_lite;MXN/GBP;Valuta;0" />');
		document.writeln('<param name="mxn_12" value="MXNHKD|idc_lite;MXN/HKD;Valuta;0" />');
		document.writeln('<param name="mxn_13" value="MXNJPY|idc_lite;MXN/JPY;Valuta;0" />');
		document.writeln('<param name="mxn_14" value="MXNPLX|idc_lite;MXN/PLX;Valuta;0" />');
		document.writeln('<param name="mxn_15" value="MXNSGD|idc_lite;MXN/SGD;Valuta;0" />');		
		document.writeln('<param name="mxn_16" value="MXNUSD|idc_lite;MXN/USD;Valuta;0" />');
		document.writeln('<param name="mxn_17" value="GBPMXN|idc_lite;GBP/MXN;Valuta;0" />');
		document.writeln('<param name="mxn_18" value="MXNXCU|idc_lite;MXN/XCU;Valuta;0" />');
		document.writeln('<param name="mxn_19" value="MXNZAR|idc_lite;MXN/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_177_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_178" value="@HTML|mad_|Moroccan Dirham|false|true|false|false" />');
		document.writeln('<param name="mad_1" value="MADEUR|idc_lite;MAD/EUR;Valuta;0" />');
		document.writeln('<param name="mad_2" value="MADGBP|idc_lite;MAD/GBP;Valuta;0" />');
		document.writeln('<param name="mad_3" value="MADUSD|idc_lite;MAD/USD;Valuta;0" />');
		document.writeln('<param name="myr_4" value="AEDMAD|idc_lite;AED/MAD;Valuta;0" />');
		document.writeln('<param name="myr_5" value="AUDMAD|idc_lite;AUD/MAD;Valuta;0" />');
		document.writeln('<param name="myr_6" value="CADMAD|idc_lite;CAD/MAD;Valuta;0" />');
		document.writeln('<param name="myr_7" value="CHFMAD|idc_lite;CHF/MAD;Valuta;0" />');
		document.writeln('<param name="myr_8" value="EURMAD|idc_lite;EUR/MAD;Valuta;0" />');
		document.writeln('<param name="myr_9" value="GBPMAD|idc_lite;GBP/MAD;Valuta;0" />');
		document.writeln('<param name="myr_10" value="JPYMAD|idc_lite;JPY/MAD;Valuta;0" />');
		document.writeln('<param name="myr_11" value="USDMAD|idc_lite;USD/MAD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_178_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_179" value="@HTML|mzn_|Mozambican Metical|false|true|false|false" />');
		document.writeln('<param name="mzn_1" value="MZNUSD|idc_lite;MZN/USD;Valuta;0" />');
		document.writeln('<param name="mzn_2" value="CADMZN|idc_lite;CAD/MZN;Valuta;0" />');
		document.writeln('<param name="mzn_3" value="CHFMZN|idc_lite;CHF/MZN;Valuta;0" />');
		document.writeln('<param name="mzn_4" value="EURMZN|idc_lite;EUR/MZN;Valuta;0" />');
		document.writeln('<param name="mzn_5" value="GBPMZN|idc_lite;GBP/MZN;Valuta;0" />');
		document.writeln('<param name="mzn_6" value="JPYMZN|idc_lite;JPY/MZN;Valuta;0" />');
		document.writeln('<param name="mzn_7" value="USDMZN|idc_lite;USD/MZN;Valuta;0" />');
		document.writeln('<param name="iexplore_market_179_path" value="Cross Rates" />');		
		
		document.writeln('<param name="iexplore_market_180" value="@HTML|nad_|Namibian Dollar|false|true|false|false" />');
		document.writeln('<param name="nad_1" value="NADEUR|idc_lite;NAD/EUR;Valuta;0" />');
		document.writeln('<param name="nad_2" value="CADNAD|idc_lite;CAD/NAD;Valuta;0" />');
		document.writeln('<param name="nad_3" value="CHFNAD|idc_lite;CHF/NAD;Valuta;0" />');
		document.writeln('<param name="nad_4" value="EURNAD|idc_lite;EUR/NAD;Valuta;0" />');
		document.writeln('<param name="nad_5" value="GBPNAD|idc_lite;GBP/NAD;Valuta;0" />');
		document.writeln('<param name="nad_6" value="JPYNAD|idc_lite;JPY/NAD;Valuta;0" />');
		document.writeln('<param name="nad_7" value="USDNAD|idc_lite;USD/NAD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_180_path" value="Cross Rates" />');	

		document.writeln('<param name="iexplore_market_181" value="@HTML|ang_|Netherlands Antilles Guilder|false|true|false|false" />');
		document.writeln('<param name="ang_1" value="ANGEUR|idc_lite;ANG/EUR;Valuta;0" />');
		document.writeln('<param name="ang_2" value="ANGSRD|idc_lite;ANG/SRD;Valuta;0" />');
		document.writeln('<param name="ang_3" value="ANGZAR|idc_lite;ANG/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_181_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_182" value="@HTML|nzd_|New Zealand Dollar|false|true|false|false" />');
		document.writeln('<param name="nzd_1" value="AUDNZD|idc_lite;AUD/NZD;Valuta;0" />');
		document.writeln('<param name="nzd_2" value="EURNZD|idc_lite;EUR/NZD;Valuta;0" />');
		document.writeln('<param name="nzd_3" value="GBPNZD|idc_lite;GBP/NZD;Valuta;0" />');
		document.writeln('<param name="nzd_4" value="NZDCHF|idc_lite;NZD/CHF;Valuta;0" />');
		document.writeln('<param name="nzd_5" value="NZDJPY|idc_lite;NZD/JPY;Valuta;0" />');
		document.writeln('<param name="nzd_6" value="NZDUSD|idc_lite;NZD/USD;Valuta;0" />');
		document.writeln('<param name="nzd_7" value="NZDSGD|idc_lite;NZD/SGD;Valuta;0" />');
		document.writeln('<param name="nzd_8" value="NZDAED|idc_lite;NZD/AED;Valuta;0" />');
		document.writeln('<param name="nzd_9" value="NZDAUD|idc_lite;NZD/AUD;Valuta;0" />');
		document.writeln('<param name="nzd_10" value="NZDBRL|idc_lite;NZD/BRL;Valuta;0" />');
		document.writeln('<param name="nzd_11" value="NZDCAD|idc_lite;NZD/CAD;Valuta;0" />');
		document.writeln('<param name="nzd_12" value="NZDCHF|idc_lite;NZD/CHF;Valuta;0" />');
		document.writeln('<param name="nzd_13" value="NZDCNH|idc_lite;NZD/CNH;Valuta;0" />');
		document.writeln('<param name="nzd_14" value="NZDCNY|idc_lite;NZD/CNY;Valuta;0" />');
		document.writeln('<param name="nzd_15" value="NZDCZK|idc_lite;NZD/CZK;Valuta;0" />');
		document.writeln('<param name="nzd_16" value="NZDDKK|idc_lite;NZD/DKK;Valuta;0" />');
		document.writeln('<param name="nzd_17" value="NZDEUR|idc_lite;NZD/EUR;Valuta;0" />');
		document.writeln('<param name="nzd_18" value="NZDGBP|idc_lite;NZD/GBP;Valuta;0" />');
		document.writeln('<param name="nzd_19" value="NZDHKD|idc_lite;NZD/HKD;Valuta;0" />');
		document.writeln('<param name="nzd_20" value="NZDHUF|idc_lite;NZD/HUF;Valuta;0" />');
		document.writeln('<param name="nzd_21" value="NZDIDR|idc_lite;NZD/IDR;Valuta;0" />');
		document.writeln('<param name="nzd_22" value="NZDILS|idc_lite;NZD/ILS;Valuta;0" />');
		document.writeln('<param name="nzd_23" value="NZDINR|idc_lite;NZD/INR;Valuta;0" />');
		document.writeln('<param name="nzd_24" value="NZDJPY|idc_lite;NZD/JPY;Valuta;0" />');
		document.writeln('<param name="nzd_25" value="NZDKRW|idc_lite;NZD/KRW;Valuta;0" />');
		document.writeln('<param name="nzd_26" value="NZDMXN|idc_lite;NZD/MXN;Valuta;0" />');
		document.writeln('<param name="nzd_27" value="NZDMYR|idc_lite;NZD/MYR;Valuta;0" />');
		document.writeln('<param name="nzd_28" value="NZDNOK|idc_lite;NZD/NOK;Valuta;0" />');
		document.writeln('<param name="nzd_29" value="NZDPGK|idc_lite;NZD/PGK;Valuta;0" />');
		document.writeln('<param name="nzd_30" value="NZDPHP|idc_lite;NZD/PHP;Valuta;0" />');
		document.writeln('<param name="nzd_31" value="NZDPKR|idc_lite;NZD/PKR;Valuta;0" />');
		document.writeln('<param name="nzd_32" value="NZDPLN|idc_lite;NZD/PLN;Valuta;0" />');
		document.writeln('<param name="nzd_33" value="NZDSAR|idc_lite;NZD/SAR;Valuta;0" />');
		document.writeln('<param name="nzd_34" value="NZDSEK|idc_lite;NZD/SEK;Valuta;0" />');
		document.writeln('<param name="nzd_35" value="NZDSGD|idc_lite;NZD/SGD;Valuta;0" />');
		document.writeln('<param name="nzd_36" value="NZDTHB|idc_lite;NZD/THB;Valuta;0" />');
		document.writeln('<param name="nzd_37" value="NZDTND|idc_lite;NZD/TND;Valuta;0" />');
		document.writeln('<param name="nzd_38" value="NZDTWD|idc_lite;NZD/TWD;Valuta;0" />');
		document.writeln('<param name="nzd_39" value="NZDXCU|idc_lite;NZD/XCU;Valuta;0" />');
		document.writeln('<param name="nzd_40" value="NZDZAR|idc_lite;NZD/ZAR;Valuta;0" />');	
		document.writeln('<param name="iexplore_market_182_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_183" value="@HTML|ngn_|Nigerian Naira|false|true|false|false" />');
		document.writeln('<param name="ngn_1" value="NGNEUR|idc_lite;NGN/EUR;Valuta;0" />');
		document.writeln('<param name="ngn_2" value="CADNGN|idc_lite;CAD/NGN;Valuta;0" />');
		document.writeln('<param name="ngn_3" value="CHFNGN|idc_lite;CHF/NGN;Valuta;0" />');
		document.writeln('<param name="ngn_4" value="EURNGN|idc_lite;EUR/NGN;Valuta;0" />');
		document.writeln('<param name="ngn_5" value="GBPNGN|idc_lite;GBP/NGN;Valuta;0" />');
		document.writeln('<param name="ngn_6" value="JPYNGN|idc_lite;JPY/NGN;Valuta;0" />');
		document.writeln('<param name="ngn_7" value="USDNGN|idc_lite;USD/NGN;Valuta;0" />');
		document.writeln('<param name="iexplore_market_183_path" value="Cross Rates" />');	

		document.writeln('<param name="iexplore_market_184" value="@HTML|nok_|Norwegian Krone|false|true|false|false" />');
		document.writeln('<param name="nok_1" value="CHFNOK|idc_lite;CHF/NOK;Valuta;0" />');
		document.writeln('<param name="nok_2" value="EURNOK|idc_lite;EUR/NOK;Valuta;0" />');
		document.writeln('<param name="nok_3" value="GBPNOK|idc_lite;GBP/NOK;Valuta;0" />');
		document.writeln('<param name="nok_4" value="JPYNOK|idc_lite;JPY/NOK;Valuta;0" />');
		document.writeln('<param name="nok_5" value="USDNOK|idc_lite;USD/NOK;Valuta;0" />');		
		document.writeln('<param name="nok_6" value="NOKAED|idc_lite;NOK/AED;Valuta;0" />');
		document.writeln('<param name="nok_7" value="NOKAUD|idc_lite;NOK/AUD;Valuta;0" />');
		document.writeln('<param name="nok_8" value="NOKCAD|idc_lite;NOK/CAD;Valuta;0" />');
		document.writeln('<param name="nok_9" value="NOKCAX|idc_lite;NOK/CAX;Valuta;0" />');
		document.writeln('<param name="nok_10" value="NOKCHF|idc_lite;NOK/CHF;Valuta;0" />');
		document.writeln('<param name="nok_11" value="NOKCNH|idc_lite;NOK/CNH;Valuta;0" />');
		document.writeln('<param name="nok_12" value="NOKCUC|idc_lite;NOK/CUC;Valuta;0" />');
		document.writeln('<param name="nok_13" value="NOKCZK|idc_lite;NOK/CZK;Valuta;0" />');
		document.writeln('<param name="nok_14" value="NOKDKK|idc_lite;NOK/DKK;Valuta;0" />');
		document.writeln('<param name="nok_15" value="NOKEUR|idc_lite;NOK/EUR;Valuta;0" />');
		document.writeln('<param name="nok_16" value="NOKGBP|idc_lite;NOK/GBP;Valuta;0" />');
		document.writeln('<param name="nok_17" value="NOKHKD|idc_lite;NOK/HKD;Valuta;0" />');
		document.writeln('<param name="nok_18" value="NOKJPY|idc_lite;NOK/JPY;Valuta;0" />');		
		document.writeln('<param name="nok_19" value="NOKNZD|idc_lite;NOK/NZD;Valuta;0" />');
		document.writeln('<param name="nok_20" value="NOKPKR|idc_lite;NOK/PKR;Valuta;0" />');
		document.writeln('<param name="nok_21" value="NOKRON|idc_lite;NOK/RON;Valuta;0" />');
		document.writeln('<param name="nok_22" value="NOKRSD|idc_lite;NOK/RSD;Valuta;0" />');
		document.writeln('<param name="nok_23" value="NOKSEK|idc_lite;NOK/SEK;Valuta;0" />');
		document.writeln('<param name="nok_24" value="NOKSGD|idc_lite;NOK/SGD;Valuta;0" />');
		document.writeln('<param name="nok_25" value="NOKUSD|idc_lite;NOK/USD;Valuta;0" />');
		document.writeln('<param name="nok_26" value="NOKXCU|idc_lite;NOK/XCU;Valuta;0" />');
		document.writeln('<param name="nok_27" value="NOKZAR|idc_lite;NOK/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_184_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_185" value="@HTML|omr_|Omani Rial|false|true|false|false" />');
		document.writeln('<param name="omr_1" value="OMREUR|idc_lite;OMR/EUR;Valuta;0" />');
		document.writeln('<param name="omr_2" value="OMRGBP|idc_lite;OMR/GBP;Valuta;0" />');
		document.writeln('<param name="omr_3" value="OMRPKR|idc_lite;OMR/PKR;Valuta;0" />');
		document.writeln('<param name="omr_4" value="OMRUSD|idc_lite;OMR/USD;Valuta;0" />');
		document.writeln('<param name="omr_5" value="AUDOMR|idc_lite;AUD/OMR;Valuta;0" />');
		document.writeln('<param name="omr_6" value="CADOMR|idc_lite;CAD/OMR;Valuta;0" />');
		document.writeln('<param name="omr_7" value="CHFOMR|idc_lite;CHF/OMR;Valuta;0" />');
		document.writeln('<param name="omr_8" value="EUROMR|idc_lite;EUR/OMR;Valuta;0" />');
		document.writeln('<param name="omr_9" value="GBPOMR|idc_lite;GBP/OMR;Valuta;0" />');
		document.writeln('<param name="omr_10" value="JPYOMR|idc_lite;JPY/OMR;Valuta;0" />');
		document.writeln('<param name="omr_11" value="SGDOMR|idc_lite;SGD/OMR;Valuta;0" />');
		document.writeln('<param name="omr_12" value="USDOMR|idc_lite;USD/OMR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_185_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_186" value="@HTML|pgk_|Papua New Guinean Kina|false|true|false|false" />');
		document.writeln('<param name="pgk_1" value="PGKHKD|idc_lite;PGK/HKD;Valuta;0" />');
		document.writeln('<param name="pgk_2" value="CHFPGK|idc_lite;CHF/PGK;Valuta;0" />');
		document.writeln('<param name="pgk_3" value="EURPGK|idc_lite;EUR/PGK;Valuta;0" />');
		document.writeln('<param name="pgk_4" value="NZDPGK|idc_lite;NZD/PGK;Valuta;0" />');
		document.writeln('<param name="pgk_5" value="USDPGK|idc_lite;USD/PGK;Valuta;0" />');
		document.writeln('<param name="iexplore_market_186_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_187" value="@HTML|pkr_|Pakistani Rupee|false|true|false|false" />');
		document.writeln('<param name="pkr_1" value="EURPKR|idc_lite;EUR/PKR;Valuta;0" />');
		document.writeln('<param name="pkr_2" value="GBPPKR|idc_lite;GBP/PKR;Valuta;0" />');
		document.writeln('<param name="pkr_3" value="USDPKR|idc_lite;USD/PKR;Valuta;0" />');
		document.writeln('<param name="pkr_4" value="PKRGBP|idc_lite;PKR/GBP;Valuta;0" />');
		document.writeln('<param name="pkr_5" value="PKRJPY|idc_lite;PKR/JPY;Valuta;0" />');
		document.writeln('<param name="pkr_6" value="PKRZAR|idc_lite;PKR/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_187_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_188" value="@HTML|pen_|Peruvian Nuevo Sol|false|true|false|false" />');
		document.writeln('<param name="pen_1" value="PENEUR|idc_lite;PEN/EUR;Valuta;0" />');
		document.writeln('<param name="pen_2" value="AUDPEN|idc_lite;AUD/PEN;Valuta;0" />');
		document.writeln('<param name="pen_3" value="CADPEN|idc_lite;CAD/PEN;Valuta;0" />');		
		document.writeln('<param name="pen_4" value="CHFPEN|idc_lite;CHF/PEN;Valuta;0" />');
		document.writeln('<param name="pen_5" value="EURPEN|idc_lite;EUR/PEN;Valuta;0" />');
		document.writeln('<param name="pen_6" value="GBPPEN|idc_lite;GBP/PEN;Valuta;0" />');		
		document.writeln('<param name="pen_7" value="JPYPEN|idc_lite;JPY/PEN;Valuta;0" />');
		document.writeln('<param name="pen_8" value="USDPEN|idc_lite;USD/PEN;Valuta;0" />');
		document.writeln('<param name="iexplore_market_188_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_189" value="@HTML|php_|Philippine Peso|false|true|false|false" />');
		document.writeln('<param name="php_1" value="EURPHP|idc_lite;EUR/PHP;Valuta;0" />');
		document.writeln('<param name="php_2" value="GBPPHP|idc_lite;GBP/PHP;Valuta;0" />');
		document.writeln('<param name="php_3" value="USDPHP|idc_lite;USD/PHP;Valuta;0" />');
		document.writeln('<param name="php_4" value="PHPAUD|idc_lite;PHP/AUD;Valuta;0" />');
		document.writeln('<param name="php_5" value="PHPCAD|idc_lite;PHP/CAD;Valuta;0" />');
		document.writeln('<param name="php_6" value="PHPCHF|idc_lite;PHP/CHF;Valuta;0" />');
		document.writeln('<param name="php_7" value="PHPCNY|idc_lite;PHP/CNY;Valuta;0" />');
		document.writeln('<param name="php_8" value="PHPEUR|idc_lite;PHP/EUR;Valuta;0" />');
		document.writeln('<param name="php_9" value="PHPGBP|idc_lite;PHP/GBP;Valuta;0" />');
		document.writeln('<param name="php_10" value="PHPHKD|idc_lite;PHP/HKD;Valuta;0" />');
		document.writeln('<param name="php_11" value="PHPIDR|idc_lite;PHP/IDR;Valuta;0" />');
		document.writeln('<param name="php_12" value="PHPINR|idc_lite;PHP/INR;Valuta;0" />');
		document.writeln('<param name="php_13" value="PHPJPY|idc_lite;PHP/JPY;Valuta;0" />');
		document.writeln('<param name="php_14" value="PHPKRW|idc_lite;PHP/KRW;Valuta;0" />');
		document.writeln('<param name="php_15" value="PHPMYR|idc_lite;PHP/MYR;Valuta;0" />');
		document.writeln('<param name="php_16" value="PHPNZD|idc_lite;PHP/NZD;Valuta;0" />');
		document.writeln('<param name="php_17" value="PHPSGD|idc_lite;PHP/SGD;Valuta;0" />');
		document.writeln('<param name="php_18" value="PHPTHB|idc_lite;PHP/THB;Valuta;0" />');
		document.writeln('<param name="php_19" value="PHPTWD|idc_lite;PHP/TWD;Valuta;0" />');
		document.writeln('<param name="php_20" value="PHPUSD|idc_lite;PHP/USD;Valuta;0" />');	
		document.writeln('<param name="iexplore_market_189_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_190" value="@HTML|pln_|Polish Zloty|false|true|false|false" />');
		document.writeln('<param name="pln_1" value="CHFPLN|idc_lite;CHF/PLN;Valuta;0" />');
		document.writeln('<param name="pln_2" value="EURPLN|idc_lite;EUR/PLN;Valuta;0" />');
		document.writeln('<param name="pln_3" value="GBPPLN|idc_lite;GBP/PLN;Valuta;0" />');
		document.writeln('<param name="pln_4" value="JPYPLN|idc_lite;JPY/PLN;Valuta;0" />');
		document.writeln('<param name="pln_5" value="USDPLN|idc_lite;USD/PLN;Valuta;0" />');
		document.writeln('<param name="pln_6" value="PLNCAD|idc_lite;PLN/CAD;Valuta;0" />');
		document.writeln('<param name="pln_7" value="PLNCAX|idc_lite;PLN/CAX;Valuta;0" />');
		document.writeln('<param name="pln_8" value="PLNCHF|idc_lite;PLN/CHF;Valuta;0" />');
		document.writeln('<param name="pln_9" value="PLNCZK|idc_lite;PLN/CZK;Valuta;0" />');
		document.writeln('<param name="pln_10" value="PLNDKK|idc_lite;PLN/DKK;Valuta;0" />');
		document.writeln('<param name="pln_11" value="PLNEUR|idc_lite;PLN/EUR;Valuta;0" />');
		document.writeln('<param name="pln_12" value="PLNGBP|idc_lite;PLN/GBP;Valuta;0" />');
		document.writeln('<param name="pln_13" value="PLNHUF|idc_lite;PLN/HUF;Valuta;0" />');
		document.writeln('<param name="pln_14" value="PLNJPY|idc_lite;PLN/JPY;Valuta;0" />');
		document.writeln('<param name="pln_15" value="PLNNOK|idc_lite;PLN/NOK;Valuta;0" />');
		document.writeln('<param name="pln_16" value="PLNRON|idc_lite;PLN/RON;Valuta;0" />');
		document.writeln('<param name="pln_17" value="PLNSEK|idc_lite;PLN/SEK;Valuta;0" />');
		document.writeln('<param name="pln_18" value="PLNTHB|idc_lite;PLN/THB;Valuta;0" />');
		document.writeln('<param name="pln_19" value="PLNUSD|idc_lite;PLN/USD;Valuta;0" />');
		document.writeln('<param name="pln_20" value="PLNXCU|idc_lite;PLN/XCU;Valuta;0" />');
		document.writeln('<param name="pln_21" value="PLNZAR|idc_lite;PLN/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_190_path" value="Cross Rates" />');
			
		document.writeln('<param name="iexplore_market_191" value="@HTML|qar_|Qatari Riyal|false|true|false|false" />');
		document.writeln('<param name="qar_1" value="QAREUR|idc_lite;QAR/EUR;Valuta;0" />');
		document.writeln('<param name="qar_2" value="QARGBP|idc_lite;QAR/GBP;Valuta;0" />');
		document.writeln('<param name="qar_3" value="QARJPY|idc_lite;QAR/JPY;Valuta;0" />');
		document.writeln('<param name="qar_4" value="QARPKR|idc_lite;QAR/PKR;Valuta;0" />');		
		document.writeln('<param name="qar_5" value="AUDQAR|idc_lite;AUD/QAR;Valuta;0" />');
		document.writeln('<param name="qar_6" value="CADQAR|idc_lite;CAD/QAR;Valuta;0" />');
		document.writeln('<param name="qar_7" value="CHFQAR|idc_lite;CHF/QAR;Valuta;0" />');
		document.writeln('<param name="qar_8" value="DKKQAR|idc_lite;DKK/QAR;Valuta;0" />');
		document.writeln('<param name="qar_9" value="EURQAR|idc_lite;EUR/QAR;Valuta;0" />');
		document.writeln('<param name="qar_10" value="GBPQAR|idc_lite;GBP/QAR;Valuta;0" />');
		document.writeln('<param name="qar_11" value="JPYQAR|idc_lite;JPY/QAR;Valuta;0" />');
		document.writeln('<param name="qar_12" value="SEKQAR|idc_lite;SEK/QAR;Valuta;0" />');
		document.writeln('<param name="qar_13" value="USDQAR|idc_lite;USD/QAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_191_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_192" value="@HTML|ron_|Romanian Leu|false|true|false|false" />');
		document.writeln('<param name="ron_1" value="EURRON|idc_lite;EUR/RON;Valuta;0" />');
		document.writeln('<param name="ron_2" value="GBPRON|idc_lite;GBP/RON;Valuta;0" />');
		document.writeln('<param name="ron_3" value="USDRON|idc_lite;USD/RON;Valuta;0" />');
		document.writeln('<param name="ron_4" value="RONEUR|idc_lite;RON/EUR;Valuta;0" />');
		document.writeln('<param name="ron_5" value="RONGBP|idc_lite;RON/GBP;Valuta;0" />');
		document.writeln('<param name="ron_6" value="RONSEK|idc_lite;RON/SEK;Valuta;0" />');
		document.writeln('<param name="ron_7" value="RONUSD|idc_lite;RON/USD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_192_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_193" value="@HTML|rub_|Russian Ruble|false|true|false|false" />');
		document.writeln('<param name="rub_1" value="EURRUB|idc_lite;EUR/RUB;Valuta;0" />');
		document.writeln('<param name="rub_2" value="GBPRUB|idc_lite;GBP/RUB;Valuta;0" />');
		document.writeln('<param name="rub_3" value="USDRUB|idc_lite;USD/RUB;Valuta;0" />');
		document.writeln('<param name="rub_4" value="RUBAMD|idc_lite;RUB/AMD;Valuta;0" />');
		document.writeln('<param name="rub_5" value="RUBBRL|idc_lite;RUB/BRL;Valuta;0" />');
		document.writeln('<param name="rub_6" value="RUBCHF|idc_lite;RUB/CHF;Valuta;0" />');
		document.writeln('<param name="rub_7" value="RUBDKK|idc_lite;RUB/DKK;Valuta;0" />');
		document.writeln('<param name="rub_8" value="RUBEUR|idc_lite;RUB/EUR;Valuta;0" />');
		document.writeln('<param name="rub_9" value="RUBGBP|idc_lite;RUB/GBP;Valuta;0" />');
		document.writeln('<param name="rub_10" value="RUBJPY|idc_lite;RUB/JPY;Valuta;0" />');		
		document.writeln('<param name="rub_11" value="RUBMYR|idc_lite;RUB/MYR;Valuta;0" />');
		document.writeln('<param name="rub_12" value="RUBSEK|idc_lite;RUB/SEK;Valuta;0" />');
		document.writeln('<param name="rub_13" value="RUBUSD|idc_lite;RUB/USD;Valuta;0" />');
		document.writeln('<param name="rub_14" value="RUBXCU|idc_lite;RUB/XCU;Valuta;0" />');
		document.writeln('<param name="rub_15" value="RUBZAR|idc_lite;RUB/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_193_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_194" value="@HTML|wst_|Samoan Tala|false|true|false|false" />');
		document.writeln('<param name="wst_1" value="WSTUSD|idc_lite;WST/USD;Valuta;0" />');
		document.writeln('<param name="wst_2" value="CADWST|idc_lite;CAD/WST;Valuta;0" />');
		document.writeln('<param name="wst_3" value="CHFWST|idc_lite;CHF/WST;Valuta;0" />');
		document.writeln('<param name="wst_4" value="EURWST|idc_lite;EUR/WST;Valuta;0" />');
		document.writeln('<param name="wst_5" value="GBPWST|idc_lite;GBP/WST;Valuta;0" />');
		document.writeln('<param name="wst_6" value="JPYWST|idc_lite;JPY/WST;Valuta;0" />');
		document.writeln('<param name="wst_7" value="USDWST|idc_lite;USD/WST;Valuta;0" />');
		document.writeln('<param name="iexplore_market_194_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_195" value="@HTML|std_|Sao Tomean Dobra|false|true|false|false" />');
		document.writeln('<param name="std_1" value="STDUSD|idc_lite;STD/USD;Valuta;0" />');
		document.writeln('<param name="std_2" value="CADSTD|idc_lite;CAD/STD;Valuta;0" />');
		document.writeln('<param name="std_3" value="CHFSTD|idc_lite;CHF/STD;Valuta;0" />');
		document.writeln('<param name="std_4" value="EURSTD|idc_lite;EUR/STD;Valuta;0" />');		
		document.writeln('<param name="std_5" value="GBPSTD|idc_lite;GBP/STD;Valuta;0" />');
		document.writeln('<param name="std_6" value="JPYSTD|idc_lite;JPY/STD;Valuta;0" />');		
		document.writeln('<param name="std_7" value="USDSTD|idc_lite;USD/STD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_195_path" value="Cross Rates" />');		

		document.writeln('<param name="iexplore_market_196" value="@HTML|iesar_|Saudi Arabian Riyal|false|true|false|false" />');
		document.writeln('<param name="iesar_1" value="EURSAR|idc_lite;EUR/SAR;Valuta;0" />');
		document.writeln('<param name="iesar_2" value="GBPSAR|idc_lite;GBP/SAR;Valuta;0" />');
		document.writeln('<param name="iesar_3" value="USDSAR|idc_lite;USD/SAR;Valuta;0" />');
		document.writeln('<param name="iesar_4" value="SARCHF|idc_lite;SAR/CHF;Valuta;0" />');
		document.writeln('<param name="iesar_5" value="SAREUR|idc_lite;SAR/EUR;Valuta;0" />');
		document.writeln('<param name="iesar_6" value="SARGBP|idc_lite;SAR/GBP;Valuta;0" />');
		document.writeln('<param name="iesar_7" value="SARHKD|idc_lite;SAR/HKD;Valuta;0" />');
		document.writeln('<param name="iesar_8" value="SARJPY|idc_lite;SAR/JPY;Valuta;0" />');
		document.writeln('<param name="iesar_9" value="SARPKR|idc_lite;SAR/PKR;Valuta;0" />');
		document.writeln('<param name="iesar_10" value="SARSEK|idc_lite;SAR/SEK;Valuta;0" />');
		document.writeln('<param name="iesar_11" value="SARSGD|idc_lite;SAR/SGD;Valuta;0" />');
		document.writeln('<param name="iesar_12" value="SARUSD|idc_lite;SAR/USD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_196_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_197" value="@HTML|scr_|Seychellois Rupee|false|true|false|false" />');
		document.writeln('<param name="scr_1" value="SCRGBP|idc_lite;SCR/GBP;Valuta;0" />');
		document.writeln('<param name="scr_2" value="AUDSCR|idc_lite;AUD/SCR;Valuta;0" />');
		document.writeln('<param name="scr_3" value="CADSCR|idc_lite;CAD/SCR;Valuta;0" />');
		document.writeln('<param name="scr_4" value="CHFSCR|idc_lite;CHF/SCR;Valuta;0" />');
		document.writeln('<param name="scr_5" value="EURSCR|idc_lite;EUR/SCR;Valuta;0" />');
		document.writeln('<param name="scr_6" value="GBPSCR|idc_lite;GBP/SCR;Valuta;0" />');
		document.writeln('<param name="scr_7" value="JPYSCR|idc_lite;JPY/SCR;Valuta;0" />');
		document.writeln('<param name="scr_8" value="USDSCR|idc_lite;USD/SCR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_197_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_198" value="@HTML|rsd_|Serbian Dinar|false|true|false|false" />');
		document.writeln('<param name="rsd_1" value="RSDEUR|idc_lite;RSD/EUR;Valuta;0" />');
		document.writeln('<param name="rsd_2" value="RSDUSD|idc_lite;RSD/USD;Valuta;0" />');		
		document.writeln('<param name="rsd_3" value="AUDRSD|idc_lite;AUD/RSD;Valuta;0" />');
		document.writeln('<param name="rsd_4" value="CADRSD|idc_lite;CAD/RSD;Valuta;0" />');
		document.writeln('<param name="rsd_5" value="CHFRSD|idc_lite;CHF/RSD;Valuta;0" />');
		document.writeln('<param name="rsd_6" value="DKKRSD|idc_lite;DKK/RSD;Valuta;0" />');
		document.writeln('<param name="rsd_7" value="EURRSD|idc_lite;EUR/RSD;Valuta;0" />');
		document.writeln('<param name="rsd_8" value="GBPRSD|idc_lite;GBP/RSD;Valuta;0" />');		
		document.writeln('<param name="rsd_9" value="JPYRSD|idc_lite;JPY/RSD;Valuta;0" />');
		document.writeln('<param name="rsd_10" value="KWDRSD|idc_lite;KWD/RSD;Valuta;0" />');
		document.writeln('<param name="rsd_11" value="NOKRSD|idc_lite;NOK/RSD;Valuta;0" />');		
		document.writeln('<param name="rsd_12" value="SEKRSD|idc_lite;SEK/RSD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_198_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_199" value="@HTML|sgd_|Singapore Dollar|false|true|false|false" />');
		document.writeln('<param name="sgd_1" value="EURSGD|idc_lite;EUR/SGD;Valuta;0" />');
		document.writeln('<param name="sgd_2" value="GBPSGD|idc_lite;GBP/SGD;Valuta;0" />');
		document.writeln('<param name="sgd_3" value="SGDCHF|idc_lite;SGD/CHF;Valuta;0" />');
		document.writeln('<param name="sgd_4" value="SGDJPY|idc_lite;SGD/JPY;Valuta;0" />');
		document.writeln('<param name="sgd_5" value="USDSGD|idc_lite;USD/SGD;Valuta;0" />');
		document.writeln('<param name="sgd_6" value="NZDSGD|idc_lite;NZD/SGD;Valuta;0" />');
		document.writeln('<param name="sgd_7" value="SGDARS|idc_lite;SGD/ARS;Valuta;0" />');
		document.writeln('<param name="sgd_8" value="SGDAUD|idc_lite;SGD/AUD;Valuta;0" />');
		document.writeln('<param name="sgd_9" value="SGDBGN|idc_lite;SGD/BGN;Valuta;0" />');
		document.writeln('<param name="sgd_10" value="SGDBRL|idc_lite;SGD/BRL;Valuta;0" />');
		document.writeln('<param name="sgd_11" value="SGDCHF|idc_lite;SGD/CHF;Valuta;0" />');
		document.writeln('<param name="sgd_12" value="SGDCNH|idc_lite;SGD/CNH;Valuta;0" />');
		document.writeln('<param name="sgd_13" value="SGDCNY|idc_lite;SGD/CNY;Valuta;0" />');
		document.writeln('<param name="sgd_14" value="SGDDKK|idc_lite;SGD/DKK;Valuta;0" />');
		document.writeln('<param name="sgd_15" value="SGDEUR|idc_lite;SGD/EUR;Valuta;0" />');
		document.writeln('<param name="sgd_16" value="SGDGBP|idc_lite;SGD/GBP;Valuta;0" />');
		document.writeln('<param name="sgd_17" value="SGDGHS|idc_lite;SGD/GHS;Valuta;0" />');
		document.writeln('<param name="sgd_18" value="SGDHKD|idc_lite;SGD/HKD;Valuta;0" />');
		document.writeln('<param name="sgd_19" value="SGDIDR|idc_lite;SGD/IDR;Valuta;0" />');
		document.writeln('<param name="sgd_20" value="SGDINR|idc_lite;SGD/INR;Valuta;0" />');
		document.writeln('<param name="sgd_21" value="SGDJPY|idc_lite;SGD/JPY;Valuta;0" />');
		document.writeln('<param name="sgd_22" value="SGDKRW|idc_lite;SGD/KRW;Valuta;0" />');
		document.writeln('<param name="sgd_23" value="SGDLKR|idc_lite;SGD/LKR;Valuta;0" />');
		document.writeln('<param name="sgd_24" value="SGDMYR|idc_lite;SGD/MYR;Valuta;0" />');
		document.writeln('<param name="sgd_25" value="SGDNOK|idc_lite;SGD/NOK;Valuta;0" />');
		document.writeln('<param name="sgd_26" value="SGDNZD|idc_lite;SGD/NZD;Valuta;0" />');
		document.writeln('<param name="sgd_27" value="SGDOMR|idc_lite;SGD/OMR;Valuta;0" />');
		document.writeln('<param name="sgd_28" value="SGDPHP|idc_lite;SGD/PHP;Valuta;0" />');
		document.writeln('<param name="sgd_29" value="SGDPKR|idc_lite;SGD/PKR;Valuta;0" />');
		document.writeln('<param name="sgd_30" value="SGDPLN|idc_lite;SGD/PLN;Valuta;0" />');
		document.writeln('<param name="sgd_31" value="SGDRUB|idc_lite;SGD/RUB;Valuta;0" />');
		document.writeln('<param name="sgd_32" value="SGDSEK|idc_lite;SGD/SEK;Valuta;0" />');
		document.writeln('<param name="sgd_33" value="SGDTHB|idc_lite;SGD/THB;Valuta;0" />');
		document.writeln('<param name="sgd_34" value="SGDTRY|idc_lite;SGD/TRY;Valuta;0" />');
		document.writeln('<param name="sgd_35" value="SGDTWD|idc_lite;SGD/TWD;Valuta;0" />');
		document.writeln('<param name="sgd_36" value="SGDUSD|idc_lite;SGD/USD;Valuta;0" />');
		document.writeln('<param name="sgd_37" value="SGDVND|idc_lite;SGD/VND;Valuta;0" />');
		document.writeln('<param name="sgd_38" value="SGDXCU|idc_lite;SGD/XCU;Valuta;0" />');
		document.writeln('<param name="sgd_39" value="SGDZAR|idc_lite;SGD/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_199_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_200" value="@HTML|sos_|Somali Shilling|false|true|false|false" />');
		document.writeln('<param name="sos_1" value="SOSUSD|idc_lite;SOS/USD;Valuta;0" />');
		document.writeln('<param name="sos_2" value="CADSOS|idc_lite;CAD/SOS;Valuta;0" />');
		document.writeln('<param name="sos_3" value="CHFSOS|idc_lite;CHF/SOS;Valuta;0" />');
		document.writeln('<param name="sos_4" value="EURSOS|idc_lite;EUR/SOS;Valuta;0" />');
		document.writeln('<param name="sos_5" value="GBPSOS|idc_lite;GBP/SOS;Valuta;0" />');
		document.writeln('<param name="sos_6" value="JPYSOS|idc_lite;JPY/SOS;Valuta;0" />');
		document.writeln('<param name="sos_7" value="USDSOS|idc_lite;USD/SOS;Valuta;0" />');	
		document.writeln('<param name="iexplore_market_200_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_201" value="@HTML|zar_|South African Rand|false|true|false|false" />');
		document.writeln('<param name="zar_1" value="EURZAR|idc_lite;EUR/ZAR;Valuta;0" />');
		document.writeln('<param name="zar_2" value="GBPZAR|idc_lite;GBP/ZAR;Valuta;0" />');
		document.writeln('<param name="zar_3" value="USDZAR|idc_lite;USD/ZAR;Valuta;0" />');
		document.writeln('<param name="zar_4" value="ZARJPY|idc_lite;ZAR/JPY;Valuta;0" />');
		document.writeln('<param name="zar_5" value="ZARAED|idc_lite;ZAR/AED;Valuta;0" />');
		document.writeln('<param name="zar_6" value="ZARANG|idc_lite;ZAR/ANG;Valuta;0" />');
		document.writeln('<param name="zar_7" value="ZARAUD|idc_lite;ZAR/AUD;Valuta;0" />');
		document.writeln('<param name="zar_8" value="ZARBWP|idc_lite;ZAR/BWP;Valuta;0" />');
		document.writeln('<param name="zar_9" value="ZARCAD|idc_lite;ZAR/CAD;Valuta;0" />');
		document.writeln('<param name="zar_10" value="ZARCHF|idc_lite;ZAR/CHF;Valuta;0" />');
		document.writeln('<param name="zar_11" value="ZARCNY|idc_lite;ZAR/CNY;Valuta;0" />');
		document.writeln('<param name="zar_12" value="ZARDKK|idc_lite;ZAR/DKK;Valuta;0" />');
		document.writeln('<param name="zar_13" value="ZAREGP|idc_lite;ZAR/EGP;Valuta;0" />');
		document.writeln('<param name="zar_14" value="ZAREUR|idc_lite;ZAR/EUR;Valuta;0" />');
		document.writeln('<param name="zar_15" value="ZARGBP|idc_lite;ZAR/GBP;Valuta;0" />');
		document.writeln('<param name="zar_16" value="ZARHKD|idc_lite;ZAR/HKD;Valuta;0" />');		
		document.writeln('<param name="zar_17" value="ZARIDR|idc_lite;ZAR/IDR;Valuta;0" />');
		document.writeln('<param name="zar_18" value="ZARINR|idc_lite;ZAR/INR;Valuta;0" />');
		document.writeln('<param name="zar_19" value="ZARJPY|idc_lite;ZAR/JPY;Valuta;0" />');
		document.writeln('<param name="zar_20" value="ZARKES|idc_lite;ZAR/KES;Valuta;0" />');
		document.writeln('<param name="zar_21" value="ZARKRW|idc_lite;ZAR/KRW;Valuta;0" />');
		document.writeln('<param name="zar_22" value="ZARLSL|idc_lite;ZAR/LSL;Valuta;0" />');
		document.writeln('<param name="zar_23" value="ZARMWK|idc_lite;ZAR/MWK;Valuta;0" />');
		document.writeln('<param name="zar_24" value="ZARMXN|idc_lite;ZAR/MXN;Valuta;0" />');		
		document.writeln('<param name="zar_25" value="ZARMYR|idc_lite;ZAR/MYR;Valuta;0" />');
		document.writeln('<param name="zar_26" value="ZARNOK|idc_lite;ZAR/NOK;Valuta;0" />');
		document.writeln('<param name="zar_27" value="ZARNZD|idc_lite;ZAR/NZD;Valuta;0" />');
		document.writeln('<param name="zar_28" value="ZARPHP|idc_lite;ZAR/PHP;Valuta;0" />');
		document.writeln('<param name="zar_29" value="ZARPKR|idc_lite;ZAR/PKR;Valuta;0" />');
		document.writeln('<param name="zar_30" value="ZARPLN|idc_lite;ZAR/PLN;Valuta;0" />');
		document.writeln('<param name="zar_31" value="ZARSCR|idc_lite;ZAR/SCR;Valuta;0" />');
		document.writeln('<param name="zar_32" value="ZARSEK|idc_lite;ZAR/SEK;Valuta;0" />');
		document.writeln('<param name="zar_33" value="ZARSGD|idc_lite;ZAR/SGD;Valuta;0" />');
		document.writeln('<param name="zar_34" value="ZARSZL|idc_lite;ZAR/SZL;Valuta;0" />');
		document.writeln('<param name="zar_35" value="ZARTHB|idc_lite;ZAR/THB;Valuta;0" />');
		document.writeln('<param name="zar_36" value="ZARTWD|idc_lite;ZAR/TWD;Valuta;0" />');
		document.writeln('<param name="zar_37" value="ZARUSD|idc_lite;ZAR/USD;Valuta;0" />');
		document.writeln('<param name="iexplore_market_201_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_202" value="@HTML|skr_|Sri Lankan Rupee|false|true|false|false" />');
		document.writeln('<param name="lkr_1" value="LKRGBP|idc_lite;LKR/GBP;Valuta;0" />');
		document.writeln('<param name="lkr_2" value="CADLKR|idc_lite;CAD/LKR;Valuta;0" />');
		document.writeln('<param name="lkr_3" value="CHFLKR|idc_lite;CHF/LKR;Valuta;0" />');
		document.writeln('<param name="lkr_4" value="EURLKR|idc_lite;EUR/LKR;Valuta;0" />');
		document.writeln('<param name="lkr_5" value="GBPLKR|idc_lite;GBP/LKR;Valuta;0" />');
		document.writeln('<param name="lkr_6" value="JPYLKR|idc_lite;JPY/LKR;Valuta;0" />');
		document.writeln('<param name="lkr_7" value="SGDLKR|idc_lite;SGD/LKR;Valuta;0" />');
		document.writeln('<param name="lkr_8" value="USDLKR|idc_lite;USD/LKR;Valuta;0" />');		
		document.writeln('<param name="iexplore_market_202_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_203" value="@HTML|syp_|Syrian Pound|false|true|false|false" />');
		document.writeln('<param name="syp_1" value="SYPUSD|idc_lite;SYP/USD;Valuta;0" />');
		document.writeln('<param name="syp_2" value="AUDSYP|idc_lite;AUD/SYP;Valuta;0" />');
		document.writeln('<param name="syp_3" value="CADSYP|idc_lite;CAD/SYP;Valuta;0" />');
		document.writeln('<param name="syp_4" value="CHFSYP|idc_lite;CHF/SYP;Valuta;0" />');
		document.writeln('<param name="syp_5" value="EURSYP|idc_lite;EUR/SYP;Valuta;0" />');
		document.writeln('<param name="syp_6" value="GBPSYP|idc_lite;GBP/SYP;Valuta;0" />');		
		document.writeln('<param name="syp_7" value="JPYSYP|idc_lite;JPY/SYP;Valuta;0" />');
		document.writeln('<param name="syp_8" value="USDSYP|idc_lite;USD/SYP;Valuta;0" />');
		document.writeln('<param name="iexplore_market_203_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_204" value="@HTML|sek_|Swedish Krona|false|true|false|false" />');
		document.writeln('<param name="sek_1" value="CHFSEK|idc_lite;CHF/SEK;Valuta;0" />');
		document.writeln('<param name="sek_2" value="EURSEK|idc_lite;EUR/SEK;Valuta;0" />');
		document.writeln('<param name="sek_3" value="GBPSEK|idc_lite;GBP/SEK;Valuta;0" />');
		document.writeln('<param name="sek_4" value="JPYSEK|idc_lite;JPY/SEK;Valuta;0" />');
		document.writeln('<param name="sek_5" value="USDSEK|idc_lite;USD/SEK;Valuta;0" />');
		document.writeln('<param name="sek_6" value="SEKAED|idc_lite;SEK/AED;Valuta;0" />');
		document.writeln('<param name="sek_7" value="SEKANG|idc_lite;SEK/ANG;Valuta;0" />');
		document.writeln('<param name="sek_8" value="SEKAUD|idc_lite;SEK/AUD;Valuta;0" />');
		document.writeln('<param name="sek_9" value="SEKBRL|idc_lite;SEK/BRL;Valuta;0" />');
		document.writeln('<param name="sek_10" value="SEKCAD|idc_lite;SEK/CAD;Valuta;0" />');
		document.writeln('<param name="sek_11" value="SEKCHF|idc_lite;SEK/CHF;Valuta;0" />');
		document.writeln('<param name="sek_12" value="SEKCNH|idc_lite;SEK/CNH;Valuta;0" />');
		document.writeln('<param name="sek_13" value="SEKCUC|idc_lite;SEK/CUC;Valuta;0" />');
		document.writeln('<param name="sek_14" value="SEKCZK|idc_lite;SEK/CZK;Valuta;0" />');
		document.writeln('<param name="sek_15" value="SEKDKK|idc_lite;SEK/DKK;Valuta;0" />');
		document.writeln('<param name="sek_16" value="SEKEUR|idc_lite;SEK/EUR;Valuta;0" />');
		document.writeln('<param name="sek_17" value="SEKGBP|idc_lite;SEK/GBP;Valuta;0" />');
		document.writeln('<param name="sek_18" value="SEKGIP|idc_lite;SEK/GIP;Valuta;0" />');
		document.writeln('<param name="sek_19" value="SEKHKD|idc_lite;SEK/HKD;Valuta;0" />');
		document.writeln('<param name="sek_20" value="SEKIDR|idc_lite;SEK/IDR;Valuta;0" />');
		document.writeln('<param name="sek_21" value="SEKILS|idc_lite;SEK/ILS;Valuta;0" />');
		document.writeln('<param name="sek_22" value="SEKINR|idc_lite;SEK/INR;Valuta;0" />');
		document.writeln('<param name="sek_23" value="SEKISK|idc_lite;SEK/ISK;Valuta;0" />');
		document.writeln('<param name="sek_24" value="SEKJPY|idc_lite;SEK/JPY;Valuta;0" />');
		document.writeln('<param name="sek_25" value="SEKKPW|idc_lite;SEK/KPW;Valuta;0" />');		
		document.writeln('<param name="sek_26" value="SEKNOK|idc_lite;SEK/NOK;Valuta;0" />');
		document.writeln('<param name="sek_27" value="GBPSEK|idc_lite;GBP/SEK;Valuta;0" />');
		document.writeln('<param name="sek_28" value="SEKNZD|idc_lite;SEK/NZD;Valuta;0" />');
		document.writeln('<param name="sek_29" value="SEKPKR|idc_lite;SEK/PKR;Valuta;0" />');
		document.writeln('<param name="sek_30" value="SEKPLN|idc_lite;SEK/PLN;Valuta;0" />');
		document.writeln('<param name="sek_31" value="SEKPLX|idc_lite;SEK/PLX;Valuta;0" />');
		document.writeln('<param name="sek_32" value="SEKQAR|idc_lite;SEK/QAR;Valuta;0" />');
		document.writeln('<param name="sek_33" value="SEKRON|idc_lite;SEK/RON;Valuta;0" />');
		document.writeln('<param name="sek_34" value="SEKRSD|idc_lite;SEK/RSD;Valuta;0" />');
		document.writeln('<param name="sek_35" value="SEKSAR|idc_lite;SEK/SAR;Valuta;0" />');
		document.writeln('<param name="sek_36" value="SEKSGD|idc_lite;SEK/SGD;Valuta;0" />');
		document.writeln('<param name="sek_37" value="SEKUSD|idc_lite;SEK/USD;Valuta;0" />');
		document.writeln('<param name="sek_38" value="SEKXCU|idc_lite;SEK/XCU;Valuta;0" />');
		document.writeln('<param name="sek_39" value="SEKZAR|idc_lite;SEK/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_204_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_205" value="@HTML|chf_|Swiss Franc|false|true|false|false" />');
		document.writeln('<param name="chf_1" value="CHFAED|idc_lite;CHF/AED;Valuta;0" />');
		document.writeln('<param name="chf_2" value="CHFAFN|idc_lite;CHF/AFN;Valuta;0" />');
		document.writeln('<param name="chf_3" value="CHFALL|idc_lite;CHF/ALL;Valuta;0" />');
		document.writeln('<param name="chf_4" value="CHFANG|idc_lite;CHF/ANG;Valuta;0" />');
		document.writeln('<param name="chf_5" value="CHFARS|idc_lite;CHF/ARS;Valuta;0" />');
		document.writeln('<param name="chf_6" value="CHFAUD|idc_lite;CHF/AUD;Valuta;0" />');
		document.writeln('<param name="chf_7" value="CHFAWG|idc_lite;CHF/AWG;Valuta;0" />');
		document.writeln('<param name="chf_8" value="CHFBAM|idc_lite;CHF/BAM;Valuta;0" />');
		document.writeln('<param name="chf_9" value="CHFBDT|idc_lite;CHF/BDT;Valuta;0" />');
		document.writeln('<param name="chf_10" value="CHFBGN|idc_lite;CHF/BGN;Valuta;0" />');
		document.writeln('<param name="chf_11" value="CHFBHD|idc_lite;CHF/BHD;Valuta;0" />');
		document.writeln('<param name="chf_12" value="CHFBIF|idc_lite;CHF/BIF;Valuta;0" />');
		document.writeln('<param name="chf_13" value="CHFBND|idc_lite;CHF/BND;Valuta;0" />');
		document.writeln('<param name="chf_14" value="CHFBOB|idc_lite;CHF/BOB;Valuta;0" />');
		document.writeln('<param name="chf_15" value="CHFBRL|idc_lite;CHF/BRL;Valuta;0" />');
		document.writeln('<param name="chf_16" value="CHFBSD|idc_lite;CHF/BSD;Valuta;0" />');
		document.writeln('<param name="chf_17" value="CHFBWP|idc_lite;CHF/BWP;Valuta;0" />');
		document.writeln('<param name="chf_18" value="CHFBYR|idc_lite;CHF/BYR;Valuta;0" />');
		document.writeln('<param name="chf_19" value="CHFBZD|idc_lite;CHF/BZD;Valuta;0" />');
		document.writeln('<param name="chf_20" value="CHFCAD|idc_lite;CHF/CAD;Valuta;0" />');
		document.writeln('<param name="chf_21" value="CHFCDF|idc_lite;CHF/CDF;Valuta;0" />');
		document.writeln('<param name="chf_22" value="CHFCLP|idc_lite;CHF/CLP;Valuta;0" />');
		document.writeln('<param name="chf_23" value="CHFCNH|idc_lite;CHF/CNH;Valuta;0" />');
		document.writeln('<param name="chf_24" value="CHFCNY|idc_lite;CHF/CNY;Valuta;0" />');
		document.writeln('<param name="chf_25" value="CHFCOP|idc_lite;CHF/COP;Valuta;0" />');
		document.writeln('<param name="chf_26" value="CHFCRC|idc_lite;CHF/CRC;Valuta;0" />');
		document.writeln('<param name="chf_27" value="CHFCUC|idc_lite;CHF/CUC;Valuta;0" />');
		document.writeln('<param name="chf_28" value="CHFCVE|idc_lite;CHF/CVE;Valuta;0" />');
		document.writeln('<param name="chf_29" value="CHFCZK|idc_lite;CHF/CZK;Valuta;0" />');
		document.writeln('<param name="chf_30" value="CHFDJF|idc_lite;CHF/DJF;Valuta;0" />');
		document.writeln('<param name="chf_31" value="CHFDKK|idc_lite;CHF/DKK;Valuta;0" />');
		document.writeln('<param name="chf_32" value="CHFDOP|idc_lite;CHF/DOP;Valuta;0" />');		
		document.writeln('<param name="chf_33" value="CHFDZD|idc_lite;CHF/DZD;Valuta;0" />');
		document.writeln('<param name="chf_34" value="CHFEGP|idc_lite;CHF/EGP;Valuta;0" />');
		document.writeln('<param name="chf_35" value="CHFERN|idc_lite;CHF/ERN;Valuta;0" />');
		document.writeln('<param name="chf_36" value="CHFEUR|idc_lite;CHF/EUR;Valuta;0" />');
		document.writeln('<param name="chf_37" value="CHFFKP|idc_lite;CHF/FKP;Valuta;0" />');
		document.writeln('<param name="chf_38" value="CHFGBP|idc_lite;CHF/GBP;Valuta;0" />');
		document.writeln('<param name="chf_39" value="CHFGEL|idc_lite;CHF/GEL;Valuta;0" />');
		document.writeln('<param name="chf_40" value="CHFGHS|idc_lite;CHF/GHS;Valuta;0" />');
		document.writeln('<param name="chf_41" value="CHFGIP|idc_lite;CHF/GIP;Valuta;0" />');
		document.writeln('<param name="chf_42" value="CHFGMD|idc_lite;CHF/GMD;Valuta;0" />');
		document.writeln('<param name="chf_43" value="CHFGNF|idc_lite;CHF/GNF;Valuta;0" />');
		document.writeln('<param name="chf_44" value="CHFGTQ|idc_lite;CHF/GTQ;Valuta;0" />');
		document.writeln('<param name="chf_45" value="CHFGYD|idc_lite;CHF/GYD;Valuta;0" />');
		document.writeln('<param name="chf_46" value="CHFHKD|idc_lite;CHF/HKD;Valuta;0" />');
		document.writeln('<param name="chf_47" value="CHFHNL|idc_lite;CHF/HNL;Valuta;0" />');
		document.writeln('<param name="chf_48" value="CHFHRK|idc_lite;CHF/HRK;Valuta;0" />');		
		document.writeln('<param name="chf_49" value="CHFHTG|idc_lite;CHF/HTG;Valuta;0" />');
		document.writeln('<param name="chf_50" value="CHFHUF|idc_lite;CHF/HUF;Valuta;0" />');
		document.writeln('<param name="chf_51" value="CHFIDR|idc_lite;CHF/IDR;Valuta;0" />');
		document.writeln('<param name="chf_52" value="CHFILS|idc_lite;CHF/ILS;Valuta;0" />');
		document.writeln('<param name="chf_53" value="CHFINR|idc_lite;CHF/INR;Valuta;0" />');
		document.writeln('<param name="chf_54" value="CHFIQD|idc_lite;CHF/IQD;Valuta;0" />');
		document.writeln('<param name="chf_55" value="CHFIRR|idc_lite;CHF/IRR;Valuta;0" />');
		document.writeln('<param name="chf_56" value="CHFJMD|idc_lite;CHF/JMD;Valuta;0" />');
		document.writeln('<param name="chf_57" value="CHFJOD|idc_lite;CHF/JOD;Valuta;0" />');
		document.writeln('<param name="chf_58" value="CHFJPY|idc_lite;CHF/JPY;Valuta;0" />');
		document.writeln('<param name="chf_59" value="CHFKES|idc_lite;CHF/KES;Valuta;0" />');
		document.writeln('<param name="chf_60" value="CHFKHR|idc_lite;CHF/KHR;Valuta;0" />');
		document.writeln('<param name="chf_61" value="CHFKMF|idc_lite;CHF/KMF;Valuta;0" />');
		document.writeln('<param name="chf_62" value="CHFKPW|idc_lite;CHF/KPW;Valuta;0" />');
		document.writeln('<param name="chf_63" value="CHFKRW|idc_lite;CHF/KRW;Valuta;0" />');
		document.writeln('<param name="chf_64" value="CHFKWD|idc_lite;CHF/KWD;Valuta;0" />');		
		document.writeln('<param name="chf_65" value="CHFKZT|idc_lite;CHF/KZT;Valuta;0" />');
		document.writeln('<param name="chf_66" value="CHFLAK|idc_lite;CHF/LAK;Valuta;0" />');
		document.writeln('<param name="chf_67" value="CHFLBP|idc_lite;CHF/LBP;Valuta;0" />');
		document.writeln('<param name="chf_68" value="CHFLKR|idc_lite;CHF/LKR;Valuta;0" />');
		document.writeln('<param name="chf_69" value="CHFLRD|idc_lite;CHF/LRD;Valuta;0" />');
		document.writeln('<param name="chf_70" value="CHFLSL|idc_lite;CHF/LSL;Valuta;0" />');
		document.writeln('<param name="chf_71" value="CHFLTL|idc_lite;CHF/LTL;Valuta;0" />');		
		document.writeln('<param name="chf_72" value="CHFLYD|idc_lite;CHF/LYD;Valuta;0" />');	
		document.writeln('<param name="chf_73" value="CHFMAD|idc_lite;CHF/MAD;Valuta;0" />');
		document.writeln('<param name="chf_74" value="CHFMDL|idc_lite;CHF/MDL;Valuta;0" />');
		document.writeln('<param name="chf_75" value="CHFMGA|idc_lite;CHF/MGA;Valuta;0" />');
		document.writeln('<param name="chf_76" value="CHFMKD|idc_lite;CHF/MKD;Valuta;0" />');
		document.writeln('<param name="chf_77" value="CHFMMK|idc_lite;CHF/MMK;Valuta;0" />');
		document.writeln('<param name="chf_78" value="CHFMNT|idc_lite;CHF/MNT;Valuta;0" />');
		document.writeln('<param name="chf_79" value="CHFMRO|idc_lite;CHF/MRO;Valuta;0" />');
		document.writeln('<param name="chf_80" value="CHFMUR|idc_lite;CHF/MUR;Valuta;0" />');
		document.writeln('<param name="chf_81" value="CHFMVR|idc_lite;CHF/MVR;Valuta;0" />');	
		document.writeln('<param name="chf_82" value="CHFMWK|idc_lite;CHF/MWK;Valuta;0" />');
		document.writeln('<param name="chf_83" value="CHFMXN|idc_lite;CHF/MXN;Valuta;0" />');
		document.writeln('<param name="chf_84" value="CHFMYR|idc_lite;CHF/MYR;Valuta;0" />');
		document.writeln('<param name="chf_85" value="CHFMZN|idc_lite;CHF/MZN;Valuta;0" />');
		document.writeln('<param name="chf_86" value="CHFNAD|idc_lite;CHF/NAD;Valuta;0" />');
		document.writeln('<param name="chf_87" value="CHFNGN|idc_lite;CHF/NGN;Valuta;0" />');
		document.writeln('<param name="chf_88" value="CHFNIO|idc_lite;CHF/NIO;Valuta;0" />');
		document.writeln('<param name="chf_89" value="CHFNOK|idc_lite;CHF/NOK;Valuta;0" />');
		document.writeln('<param name="chf_90" value="CHFNPR|idc_lite;CHF/NPR;Valuta;0" />');			
		document.writeln('<param name="chf_91" value="CHFNZD|idc_lite;CHF/NZD;Valuta;0" />');
		document.writeln('<param name="chf_92" value="CHFOMR|idc_lite;CHF/OMR;Valuta;0" />');
		document.writeln('<param name="chf_93" value="CHFPEN|idc_lite;CHF/PEN;Valuta;0" />');
		document.writeln('<param name="chf_94" value="CHFPGK|idc_lite;CHF/PGK;Valuta;0" />');
		document.writeln('<param name="chf_95" value="CHFPHP|idc_lite;CHF/PHP;Valuta;0" />');	
		document.writeln('<param name="chf_96" value="CHFPKR|idc_lite;CHF/PKR;Valuta;0" />');
		document.writeln('<param name="chf_97" value="CHFPLN|idc_lite;CHF/PLN;Valuta;0" />');
		document.writeln('<param name="chf_98" value="CHFPYG|idc_lite;CHF/PYG;Valuta;0" />');
		document.writeln('<param name="chf_99" value="CHFQAR|idc_lite;CHF/QAR;Valuta;0" />');
		document.writeln('<param name="chf_100" value="CHFRON|idc_lite;CHF/RON;Valuta;0" />');	
		document.writeln('<param name="chf_101" value="CHFRSD|idc_lite;CHF/RSD;Valuta;0" />');
		document.writeln('<param name="chf_102" value="CHFRUB|idc_lite;CHF/RUB;Valuta;0" />');
		document.writeln('<param name="chf_103" value="CHFRWF|idc_lite;CHF/RWF;Valuta;0" />');
		document.writeln('<param name="chf_104" value="CHFSAR|idc_lite;CHF/SAR;Valuta;0" />');
		document.writeln('<param name="chf_105" value="CHFSCR|idc_lite;CHF/SCR;Valuta;0" />');	
		document.writeln('<param name="chf_106" value="CHFSDG|idc_lite;CHF/SDG;Valuta;0" />');
		document.writeln('<param name="chf_107" value="CHFSEK|idc_lite;CHF/SEK;Valuta;0" />');
		document.writeln('<param name="chf_108" value="CHFSGD|idc_lite;CHF/SGD;Valuta;0" />');
		document.writeln('<param name="chf_109" value="CHFSHP|idc_lite;CHF/SHP;Valuta;0" />');
		document.writeln('<param name="chf_110" value="CHFSLL|idc_lite;CHF/SLL;Valuta;0" />');			
		document.writeln('<param name="chf_111" value="CHFSOS|idc_lite;CHF/SOS;Valuta;0" />');
		document.writeln('<param name="chf_112" value="CHFSRD|idc_lite;CHF/SRD;Valuta;0" />');
		document.writeln('<param name="chf_113" value="CHFSTD|idc_lite;CHF/STD;Valuta;0" />');	
		document.writeln('<param name="chf_114" value="CHFSVC|idc_lite;CHF/SVC;Valuta;0" />');
		document.writeln('<param name="chf_115" value="CHFSYP|idc_lite;CHF/SYP;Valuta;0" />');
		document.writeln('<param name="chf_116" value="CHFSZL|idc_lite;CHF/SZL;Valuta;0" />');
		document.writeln('<param name="chf_117" value="CHFTHB|idc_lite;CHF/THB;Valuta;0" />');
		document.writeln('<param name="chf_118" value="CHFTMT|idc_lite;CHF/TMT;Valuta;0" />');	
		document.writeln('<param name="chf_119" value="CHFTND|idc_lite;CHF/TND;Valuta;0" />');
		document.writeln('<param name="chf_120" value="CHFTOP|idc_lite;CHF/TOP;Valuta;0" />');
		document.writeln('<param name="chf_121" value="CHFTRY|idc_lite;CHF/TRY;Valuta;0" />');
		document.writeln('<param name="chf_122" value="CHFTTD|idc_lite;CHF/TTD;Valuta;0" />');
		document.writeln('<param name="chf_123" value="CHFTWD|idc_lite;CHF/TWD;Valuta;0" />');		
		document.writeln('<param name="chf_124" value="CHFTZS|idc_lite;CHF/TZS;Valuta;0" />');
		document.writeln('<param name="chf_125" value="CHFUAH|idc_lite;CHF/UAH;Valuta;0" />');
		document.writeln('<param name="chf_126" value="CHFUGX|idc_lite;CHF/UGX;Valuta;0" />');
		document.writeln('<param name="chf_127" value="CHFUSD|idc_lite;CHF/USD;Valuta;0" />');
		document.writeln('<param name="chf_128" value="CHFUYU|idc_lite;CHF/UYU;Valuta;0" />');			
		document.writeln('<param name="chf_129" value="CHFVEF|idc_lite;CHF/VEF;Valuta;0" />');
		document.writeln('<param name="chf_130" value="CHFVND|idc_lite;CHF/VND;Valuta;0" />');
		document.writeln('<param name="chf_131" value="CHFVUV|idc_lite;CHF/VUV;Valuta;0" />');
		document.writeln('<param name="chf_132" value="CHFWST|idc_lite;CHF/WST;Valuta;0" />');
		document.writeln('<param name="chf_133" value="CHFXAF|idc_lite;CHF/XAF;Valuta;0" />');			
		document.writeln('<param name="chf_134" value="CHFXCU|idc_lite;CHF/XCU;Valuta;0" />');
		document.writeln('<param name="chf_135" value="CHFXOF|idc_lite;CHF/XOF;Valuta;0" />');
		document.writeln('<param name="chf_136" value="CHFXPF|idc_lite;CHF/XPF;Valuta;0" />');
		document.writeln('<param name="chf_137" value="CHFYER|idc_lite;CHF/YER;Valuta;0" />');
		document.writeln('<param name="chf_138" value="CHFZAR|idc_lite;CHF/ZAR;Valuta;0" />');			
		document.writeln('<param name="chf_139" value="CHFZMW|idc_lite;CHF/ZMW;Valuta;0" />');
		document.writeln('<param name="chf_140" value="CHFZWL|idc_lite;CHF/ZWL;Valuta;0" />');				
		document.writeln('<param name="iexplore_market_205_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_206" value="@HTML|twd_|Taiwan Dollar|false|true|false|false" />');
		document.writeln('<param name="twd_1" value="EURTWD|idc_lite;EUR/TWD;Valuta;0" />');
		document.writeln('<param name="twd_2" value="GBPTWD|idc_lite;GBP/TWD;Valuta;0" />');
		document.writeln('<param name="twd_3" value="USDTWD|idc_lite;USD/TWD;Valuta;0" />');		
		document.writeln('<param name="twd_4" value="TWDAUD|idc_lite;TWD/AUD;Valuta;0" />');
		document.writeln('<param name="twd_5" value="TWDCAD|idc_lite;TWD/CAD;Valuta;0" />');
		document.writeln('<param name="twd_6" value="TWDCHF|idc_lite;TWD/CHF;Valuta;0" />');
		document.writeln('<param name="twd_7" value="TWDCNH|idc_lite;TWD/CNH;Valuta;0" />');
		document.writeln('<param name="twd_8" value="TWDCNY|idc_lite;TWD/CNY;Valuta;0" />');
		document.writeln('<param name="twd_9" value="TWDEUR|idc_lite;TWD/EUR;Valuta;0" />');
		document.writeln('<param name="twd_10" value="TWDGBP|idc_lite;TWD/GBP;Valuta;0" />');
		document.writeln('<param name="twd_11" value="TWDHKD|idc_lite;TWD/HKD;Valuta;0" />');
		document.writeln('<param name="twd_11" value="TWDIDR|idc_lite;TWD/IDR;Valuta;0" />');
		document.writeln('<param name="twd_12" value="TWDINR|idc_lite;TWD/INR;Valuta;0" />');
		document.writeln('<param name="twd_13" value="TWDJPY|idc_lite;TWD/JPY;Valuta;0" />');
		document.writeln('<param name="twd_14" value="TWDKRW|idc_lite;TWD/KRW;Valuta;0" />');
		document.writeln('<param name="twd_15" value="TWDMYR|idc_lite;TWD/MYR;Valuta;0" />');
		document.writeln('<param name="twd_16" value="TWDNZD|idc_lite;TWD/NZD;Valuta;0" />');
		document.writeln('<param name="twd_17" value="TWDPHP|idc_lite;TWD/PHP;Valuta;0" />');
		document.writeln('<param name="twd_18" value="TWDPKR|idc_lite;TWD/PKR;Valuta;0" />');
		document.writeln('<param name="twd_19" value="TWDSEK|idc_lite;TWD/SEK;Valuta;0" />');
		document.writeln('<param name="twd_20" value="TWDSGD|idc_lite;TWD/SGD;Valuta;0" />');
		document.writeln('<param name="twd_21" value="TWDTHB|idc_lite;TWD/THB;Valuta;0" />');
		document.writeln('<param name="twd_22" value="TWDUSD|idc_lite;TWD/USD;Valuta;0" />');
		document.writeln('<param name="twd_23" value="TWDZAR|idc_lite;TWD/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_206_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_207" value="@HTML|tzs_|Tanzanian Shilling|false|true|false|false" />');
		document.writeln('<param name="tzs_1" value="TZSEUR|idc_lite;TZS/EUR;Valuta;0" />');
		document.writeln('<param name="tzs_2" value="CADTZS|idc_lite;CAD/TZS;Valuta;0" />');
		document.writeln('<param name="tzs_3" value="CHFTZS|idc_lite;CHF/TZS;Valuta;0" />');
		document.writeln('<param name="tzs_4" value="EURTZS|idc_lite;EUR/TZS;Valuta;0" />');
		document.writeln('<param name="tzs_5" value="GBPTZS|idc_lite;GBP/TZS;Valuta;0" />');
		document.writeln('<param name="tzs_6" value="JPYTZS|idc_lite;JPY/TZS;Valuta;0" />');
		document.writeln('<param name="tzs_7" value="USDTZS|idc_lite;USD/TZS;Valuta;0" />');	
		document.writeln('<param name="iexplore_market_207_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_208" value="@HTML|thb_|Thailand Bath|false|true|false|false" />');
		document.writeln('<param name="thb_1" value="EURTHB|idc_lite;EUR/THB;Valuta;0" />');
		document.writeln('<param name="thb_2" value="GBPTHB|idc_lite;GBP/THB;Valuta;0" />');
		document.writeln('<param name="thb_3" value="USDTHB|idc_lite;USD/THB;Valuta;0" />');
		document.writeln('<param name="thb_4" value="THBAUD|idc_lite;THB/AUD;Valuta;0" />');
		document.writeln('<param name="thb_5" value="THBCAD|idc_lite;THB/CAD;Valuta;0" />');
		document.writeln('<param name="thb_6" value="THBCHF|idc_lite;THB/CHF;Valuta;0" />');
		document.writeln('<param name="thb_7" value="THBCNY|idc_lite;THB/CNY;Valuta;0" />');
		document.writeln('<param name="thb_8" value="THBDKK|idc_lite;THB/DKK;Valuta;0" />');
		document.writeln('<param name="thb_9" value="THBEUR|idc_lite;THB/EUR;Valuta;0" />');
		document.writeln('<param name="thb_10" value="THBGBP|idc_lite;THB/GBP;Valuta;0" />');
		document.writeln('<param name="thb_11" value="THBHKD|idc_lite;THB/HKD;Valuta;0" />');
		document.writeln('<param name="thb_12" value="THBIDR|idc_lite;THB/IDR;Valuta;0" />');
		document.writeln('<param name="thb_13" value="THBINR|idc_lite;THB/INR;Valuta;0" />');
		document.writeln('<param name="thb_14" value="THBJPY|idc_lite;THB/JPY;Valuta;0" />');
		document.writeln('<param name="thb_15" value="THBKRW|idc_lite;THB/KRW;Valuta;0" />');
		document.writeln('<param name="thb_16" value="THBMYR|idc_lite;THB/MYR;Valuta;0" />');
		document.writeln('<param name="thb_17" value="THBNOK|idc_lite;THB/NOK;Valuta;0" />');
		document.writeln('<param name="thb_18" value="THBNZD|idc_lite;THB/NZD;Valuta;0" />');
		document.writeln('<param name="thb_19" value="THBPHP|idc_lite;THB/PHP;Valuta;0" />');
		document.writeln('<param name="thb_20" value="THBPKR|idc_lite;THB/PKR;Valuta;0" />');
		document.writeln('<param name="thb_21" value="THBSEK|idc_lite;THB/SEK;Valuta;0" />');
		document.writeln('<param name="thb_22" value="THBSGD|idc_lite;THB/SGD;Valuta;0" />');
		document.writeln('<param name="thb_23" value="THBTND|idc_lite;THB/TND;Valuta;0" />');
		document.writeln('<param name="thb_24" value="THBTWD|idc_lite;THB/TWD;Valuta;0" />');
		document.writeln('<param name="thb_25" value="THBUSD|idc_lite;THB/USD;Valuta;0" />');
		document.writeln('<param name="thb_26" value="THBZAR|idc_lite;THB/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_208_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_209" value="@HTML|top_|Tongan Pa\'anga|false|true|false|false" />');
		document.writeln('<param name="top_1" value="TOPUSD|idc_lite;TOP/USD;Valuta;0" />');
		document.writeln('<param name="top_2" value="CADTOP|idc_lite;CAD/TOP;Valuta;0" />');
		document.writeln('<param name="top_3" value="CHFTOP|idc_lite;CHF/TOP;Valuta;0" />');
		document.writeln('<param name="top_4" value="EURTOP|idc_lite;EUR/TOP;Valuta;0" />');
		document.writeln('<param name="top_5" value="GBPTOP|idc_lite;GBP/TOP;Valuta;0" />');
		document.writeln('<param name="top_6" value="JPYTOP|idc_lite;JPY/TOP;Valuta;0" />');
		document.writeln('<param name="top_7" value="USDTOP|idc_lite;USD/TOP;Valuta;0" />');
		document.writeln('<param name="iexplore_market_209_path" value="Cross Rates" />');	
	
		document.writeln('<param name="iexplore_market_210" value="@HTML|tnd_|Tunisian Dinar|false|true|false|false" />');
		document.writeln('<param name="tnd_1" value="TNDEUR|idc_lite;TND/EUR;Valuta;0" />');
		document.writeln('<param name="tnd_2" value="TNDGBP|idc_lite;TND/GBP;Valuta;0" />');		
		document.writeln('<param name="tnd_3" value="AUDTND|idc_lite;AUD/TND;Valuta;0" />');
		document.writeln('<param name="tnd_4" value="CHFTND|idc_lite;CHF/TND;Valuta;0" />');
		document.writeln('<param name="tnd_5" value="EURTND|idc_lite;EUR/TND;Valuta;0" />');
		document.writeln('<param name="tnd_6" value="GBPTND|idc_lite;GBP/TND;Valuta;0" />');
		document.writeln('<param name="tnd_7" value="JPYTND|idc_lite;JPY/TND;Valuta;0" />');
		document.writeln('<param name="tnd_8" value="MYRTND|idc_lite;MYR/TND;Valuta;0" />');
		document.writeln('<param name="tnd_9" value="NZDTND|idc_lite;NZD/TND;Valuta;0" />');
		document.writeln('<param name="tnd_10" value="THBTND|idc_lite;THB/TND;Valuta;0" />');
		document.writeln('<param name="tnd_11" value="USDTND|idc_lite;USD/TND;Valuta;0" />');
		document.writeln('<param name="iexplore_market_210_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_211" value="@HTML|try_|Turkish Lira|false|true|false|false" />');
		document.writeln('<param name="try_1" value="EURTRY|idc_lite;EUR/TRY;Valuta;0" />');
		document.writeln('<param name="try_2" value="GBPTRY|idc_lite;GBP/TRY;Valuta;0" />');
		document.writeln('<param name="try_3" value="USDTRY|idc_lite;USD/TRY;Valuta;0" />');
		document.writeln('<param name="try_4" value="TRYCHF|idc_lite;TRY/CHF;Valuta;0" />');
		document.writeln('<param name="try_5" value="TRYDKK|idc_lite;TRY/DKK;Valuta;0" />');
		document.writeln('<param name="try_6" value="TRYEUR|idc_lite;TRY/EUR;Valuta;0" />');
		document.writeln('<param name="try_7" value="TRYGBP|idc_lite;TRY/GBP;Valuta;0" />');
		document.writeln('<param name="try_8" value="TRYJPY|idc_lite;TRY/JPY;Valuta;0" />');
		document.writeln('<param name="try_9" value="TRYPLX|idc_lite;TRY/PLX;Valuta;0" />');
		document.writeln('<param name="try_10" value="TRYXCU|idc_lite;TRY/XCU;Valuta;0" />');
		document.writeln('<param name="iexplore_market_211_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_212" value="@HTML|aed_|UAE Dirham|false|true|false|false" />');
		document.writeln('<param name="aed_1" value="EURAED|idc_lite;EUR/AED;Valuta;0" />');
		document.writeln('<param name="aed_2" value="GBPAED|idc_lite;GBP/AED;Valuta;0" />');
		document.writeln('<param name="aed_3" value="USDAED|idc_lite;USD/AED;Valuta;0" />');
		document.writeln('<param name="aed_4" value="AEDAUD|idc_lite;AED/AUD;Valuta;0" />');
		document.writeln('<param name="aed_5" value="AEDCHF|idc_lite;AED/CHF;Valuta;0" />');
		document.writeln('<param name="aed_6" value="AEDEUR|idc_lite;AED/EUR;Valuta;0" />');
		document.writeln('<param name="aed_7" value="AEDGBP|idc_lite;AED/GBP;Valuta;0" />');
		document.writeln('<param name="aed_8" value="AEDHKD|idc_lite;AED/HKD;Valuta;0" />');
		document.writeln('<param name="aed_9" value="AEDJPY|idc_lite;AED/JPY;Valuta;0" />');
		document.writeln('<param name="aed_10" value="AEDMAD|idc_lite;AED/MAD;Valuta;0" />');
		document.writeln('<param name="aed_11" value="AEDMUR|idc_lite;AED/MUR;Valuta;0" />');
		document.writeln('<param name="aed_12" value="AEDNOK|idc_lite;AED/NOK;Valuta;0" />');
		document.writeln('<param name="aed_13" value="AEDNZD|idc_lite;AED/NZD;Valuta;0" />');
		document.writeln('<param name="aed_14" value="AEDPKR|idc_lite;AED/PKR;Valuta;0" />');
		document.writeln('<param name="aed_15" value="AEDSAR|idc_lite;AED/SAR;Valuta;0" />');
		document.writeln('<param name="aed_16" value="AEDSEK|idc_lite;AED/SEK;Valuta;0" />');
		document.writeln('<param name="aed_17" value="AEDSGD|idc_lite;AED/SGD;Valuta;0" />');
		document.writeln('<param name="aed_18" value="AEDTRY|idc_lite;AED/TRY;Valuta;0" />');
		document.writeln('<param name="aed_19" value="AEDUSD|idc_lite;AED/USD;Valuta;0" />');
		document.writeln('<param name="aed_20" value="AEDZAR|idc_lite;AED/ZAR;Valuta;0" />');
		document.writeln('<param name="iexplore_market_212_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_213" value="@HTML|uah_|Ukraine Karbovanet|false|true|false|false" />');
		document.writeln('<param name="uah_1" value="EURUAH|idc_lite;EUR/UAH;Valuta;0" />');
		document.writeln('<param name="uah_2" value="GBPUAH|idc_lite;GBP/UAH;Valuta;0" />');
		document.writeln('<param name="uah_3" value="USDUAH|idc_lite;USD/UAH;Valuta;0" />');
		document.writeln('<param name="iexplore_market_213_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_214" value="@HTML|gbp_|United Kingdom Pound|false|true|false|false" />');
		document.writeln('<param name="gbp_1" value="GBPAED|idc_lite;GBP/AED;Valuta;0" />');
		document.writeln('<param name="gbp_2" value="GBPAFN|idc_lite;GBP/AFN;Valuta;0" />');
		document.writeln('<param name="gbp_3" value="GBPANG|idc_lite;GBP/ANG;Valuta;0" />');
		document.writeln('<param name="gbp_4" value="GBPARS|idc_lite;GBP/ARS;Valuta;0" />');
		document.writeln('<param name="gbp_5" value="GBPAUD|idc_lite;GBP/AUD;Valuta;0" />');
		document.writeln('<param name="gbp_6" value="GBPAWG|idc_lite;GBP/AWG;Valuta;0" />');
		document.writeln('<param name="gbp_7" value="GBPAZN|idc_lite;GBP/AZN;Valuta;0" />');
		document.writeln('<param name="gbp_8" value="GBPBAM|idc_lite;GBP/BAM;Valuta;0" />');
		document.writeln('<param name="gbp_9" value="GBPBBD|idc_lite;GBP/BBD;Valuta;0" />');
		document.writeln('<param name="gbp_10" value="GBPBDT|idc_lite;GBP/BDT;Valuta;0" />');
		document.writeln('<param name="gbp_11" value="GBPBGN|idc_lite;GBP/BGN;Valuta;0" />');
		document.writeln('<param name="gbp_12" value="GBPBHD|idc_lite;GBP/BHD;Valuta;0" />');
		document.writeln('<param name="gbp_13" value="GBPBIF|idc_lite;GBP/BIF;Valuta;0" />');
		document.writeln('<param name="gbp_14" value="GBPBMD|idc_lite;GBP/BMD;Valuta;0" />');
		document.writeln('<param name="gbp_15" value="GBPBND|idc_lite;GBP/BND;Valuta;0" />');
		document.writeln('<param name="gbp_16" value="GBPBOB|idc_lite;GBP/BOB;Valuta;0" />');
		document.writeln('<param name="gbp_17" value="GBPBRL|idc_lite;GBP/BRL;Valuta;0" />');
		document.writeln('<param name="gbp_18" value="GBPBSD|idc_lite;GBP/BSD;Valuta;0" />');
		document.writeln('<param name="gbp_19" value="GBPBWP|idc_lite;GBP/BWP;Valuta;0" />');
		document.writeln('<param name="gbp_20" value="GBPBYR|idc_lite;GBP/BYR;Valuta;0" />');
		document.writeln('<param name="gbp_21" value="GBPBZD|idc_lite;GBP/BZD;Valuta;0" />');
		document.writeln('<param name="gbp_22" value="GBPCAD|idc_lite;GBP/CAD;Valuta;0" />');
		document.writeln('<param name="gbp_23" value="GBPCAX|idc_lite;GBP/CAX;Valuta;0" />');
		document.writeln('<param name="gbp_24" value="GBPCDF|idc_lite;GBP/CDF;Valuta;0" />');
		document.writeln('<param name="gbp_25" value="GBPCHF|idc_lite;GBP/CHF;Valuta;0" />');
		document.writeln('<param name="gbp_26" value="GBPCLP|idc_lite;GBP/CLP;Valuta;0" />');
		document.writeln('<param name="gbp_27" value="GBPCNH|idc_lite;GBP/CNH;Valuta;0" />');
		document.writeln('<param name="gbp_28" value="GBPCNY|idc_lite;GBP/CNY;Valuta;0" />');
		document.writeln('<param name="gbp_29" value="GBPCOP|idc_lite;GBP/COP;Valuta;0" />');
		document.writeln('<param name="gbp_30" value="GBPCRC|idc_lite;GBP/CRC;Valuta;0" />');
		document.writeln('<param name="gbp_31" value="GBPCUC|idc_lite;GBP/CUC;Valuta;0" />');
		document.writeln('<param name="gbp_32" value="GBPCUP|idc_lite;GBP/CUP;Valuta;0" />');
		document.writeln('<param name="gbp_33" value="GBPCVE|idc_lite;GBP/CVE;Valuta;0" />');
		document.writeln('<param name="gbp_34" value="GBPCZK|idc_lite;GBP/CZK;Valuta;0" />');
		document.writeln('<param name="gbp_35" value="GBPDJF|idc_lite;GBP/DJF;Valuta;0" />');
		document.writeln('<param name="gbp_36" value="GBPDKK|idc_lite;GBP/DKK;Valuta;0" />');
		document.writeln('<param name="gbp_37" value="GBPDOP|idc_lite;GBP/DOP;Valuta;0" />');
		document.writeln('<param name="gbp_38" value="GBPDZD|idc_lite;GBP/DZD;Valuta;0" />');
		document.writeln('<param name="gbp_39" value="GBPEGP|idc_lite;GBP/EGP;Valuta;0" />');
		document.writeln('<param name="gbp_40" value="GBPERN|idc_lite;GBP/ERN;Valuta;0" />');
		document.writeln('<param name="gbp_41" value="GBPEUR|idc_lite;GBP/EUR;Valuta;0" />');
		document.writeln('<param name="gbp_42" value="GBPFKP|idc_lite;GBP/FKP;Valuta;0" />');
		document.writeln('<param name="gbp_43" value="GBPGBP|idc_lite;GBP/GBP;Valuta;0" />');
		document.writeln('<param name="gbp_44" value="GBPGEL|idc_lite;GBP/GEL;Valuta;0" />');
		document.writeln('<param name="gbp_45" value="GBPGHS|idc_lite;GBP/GHS;Valuta;0" />');
		document.writeln('<param name="gbp_46" value="GBPGIP|idc_lite;GBP/GIP;Valuta;0" />');
		document.writeln('<param name="gbp_47" value="GBPGMD|idc_lite;GBP/GMD;Valuta;0" />');
		document.writeln('<param name="gbp_48" value="GBPGNF|idc_lite;GBP/GNF;Valuta;0" />');
		document.writeln('<param name="gbp_49" value="GBPGTQ|idc_lite;GBP/GTQ;Valuta;0" />');
		document.writeln('<param name="gbp_50" value="GBPGYD|idc_lite;GBPG/YD;Valuta;0" />');
		document.writeln('<param name="gbp_51" value="GBPHKD|idc_lite;GBP/HKD;Valuta;0" />');
		document.writeln('<param name="gbp_52" value="GBPHNL|idc_lite;GBP/HNL;Valuta;0" />');
		document.writeln('<param name="gbp_53" value="GBPHRK|idc_lite;GBP/HRK;Valuta;0" />');
		document.writeln('<param name="gbp_54" value="GBPHTG|idc_lite;GBP/HTG;Valuta;0" />');
		document.writeln('<param name="gbp_55" value="GBPHUF|idc_lite;GBP/HUF;Valuta;0" />');
		document.writeln('<param name="gbp_56" value="GBPIDR|idc_lite;GBP/IDR;Valuta;0" />');
		document.writeln('<param name="gbp_57" value="GBPILS|idc_lite;GBP/ILS;Valuta;0" />');
		document.writeln('<param name="gbp_58" value="GBPINR|idc_lite;GBP/INR;Valuta;0" />');
		document.writeln('<param name="gbp_59" value="GBPIQD|idc_lite;GBP/IQD;Valuta;0" />');
		document.writeln('<param name="gbp_60" value="GBPIRR|idc_lite;GBP/IRR;Valuta;0" />');
		document.writeln('<param name="gbp_61" value="GBPJMD|idc_lite;GBP/JMD;Valuta;0" />');
		document.writeln('<param name="gbp_62" value="GBPJOD|idc_lite;GBP/JOD;Valuta;0" />');
		document.writeln('<param name="gbp_63" value="GBPJPY|idc_lite;GBP/JPY;Valuta;0" />');
		document.writeln('<param name="gbp_64" value="GBPKES|idc_lite;GBP/KES;Valuta;0" />');
		document.writeln('<param name="gbp_65" value="GBPKGS|idc_lite;GBP/KGS;Valuta;0" />');
		document.writeln('<param name="gbp_66" value="GBPKHR|idc_lite;GBP/KHR;Valuta;0" />');
		document.writeln('<param name="gbp_67" value="GBPKMF|idc_lite;GBP/KMF;Valuta;0" />');
		document.writeln('<param name="gbp_68" value="GBPKPW|idc_lite;GBP/KPW;Valuta;0" />');
		document.writeln('<param name="gbp_69" value="GBPKRW|idc_lite;GBP/KRW;Valuta;0" />');
		document.writeln('<param name="gbp_70" value="GBPKWD|idc_lite;GBP/KWD;Valuta;0" />');
		document.writeln('<param name="gbp_71" value="GBPKYD|idc_lite;GBP/KYD;Valuta;0" />');
		document.writeln('<param name="gbp_72" value="GBPKZT|idc_lite;GBP/KZT;Valuta;0" />');
		document.writeln('<param name="gbp_73" value="GBPLAK|idc_lite;GBP/LAK;Valuta;0" />');
		document.writeln('<param name="gbp_74" value="GBPLBP|idc_lite;GBP/LBP;Valuta;0" />');
		document.writeln('<param name="gbp_75" value="GBPLKR|idc_lite;GBP/LKR;Valuta;0" />');
		document.writeln('<param name="gbp_76" value="GBPLRD|idc_lite;GBP/LRD;Valuta;0" />');
		document.writeln('<param name="gbp_77" value="GBPLSL|idc_lite;GBP/LSL;Valuta;0" />');
		document.writeln('<param name="gbp_78" value="GBPLTL|idc_lite;GBP/LTL;Valuta;0" />');		
		document.writeln('<param name="gbp_79" value="GBPLYD|idc_lite;GBP/LYD;Valuta;0" />');
		document.writeln('<param name="gbp_80" value="GBPMAD|idc_lite;GBP/MAD;Valuta;0" />');
		document.writeln('<param name="gbp_81" value="GBPMDL|idc_lite;GBP/MDL;Valuta;0" />');
		document.writeln('<param name="gbp_82" value="GBPMGA|idc_lite;GBP/MGA;Valuta;0" />');		
		document.writeln('<param name="gbp_83" value="GBPMKD|idc_lite;GBP/MKD;Valuta;0" />');
		document.writeln('<param name="gbp_84" value="GBPMMK|idc_lite;GBP/MMK;Valuta;0" />');
		document.writeln('<param name="gbp_85" value="GBPMNT|idc_lite;GBP/MNT;Valuta;0" />');
		document.writeln('<param name="gbp_86" value="GBPMRO|idc_lite;GBP/MRO;Valuta;0" />');
		document.writeln('<param name="gbp_87" value="GBPMUR|idc_lite;GBP/MUR;Valuta;0" />');
		document.writeln('<param name="gbp_88" value="GBPMVR|idc_lite;GBP/MVR;Valuta;0" />');
		document.writeln('<param name="gbp_89" value="GBPMWK|idc_lite;GBP/MWK;Valuta;0" />');
		document.writeln('<param name="gbp_90" value="GBPMXN|idc_lite;GBP/MXN;Valuta;0" />');
		document.writeln('<param name="gbp_91" value="GBPMYR|idc_lite;GBP/MYR;Valuta;0" />');
		document.writeln('<param name="gbp_92" value="GBPMZN|idc_lite;GBP/MZN;Valuta;0" />');
		document.writeln('<param name="gbp_93" value="GBPNAD|idc_lite;GBP/NAD;Valuta;0" />');
		document.writeln('<param name="gbp_94" value="GBPNGN|idc_lite;GBP/NGN;Valuta;0" />');
		document.writeln('<param name="gbp_95" value="GBPNIO|idc_lite;GBP/NIO;Valuta;0" />');
		document.writeln('<param name="gbp_96" value="GBPNOK|idc_lite;GBP/NOK;Valuta;0" />');
		document.writeln('<param name="gbp_97" value="GBPNPR|idc_lite;GBP/NPR;Valuta;0" />');
		document.writeln('<param name="gbp_98" value="GBPNZD|idc_lite;GBP/NZD;Valuta;0" />');
		document.writeln('<param name="gbp_99" value="GBPOMR|idc_lite;GBP/OMR;Valuta;0" />');
		document.writeln('<param name="gbp_100" value="GBPPAB|idc_lite;GBP/PAB;Valuta;0" />');
		document.writeln('<param name="gbp_101" value="GBPPEN|idc_lite;GBP/PEN;Valuta;0" />');
		document.writeln('<param name="gbp_102" value="GBPPHP|idc_lite;GBP/PHP;Valuta;0" />');
		document.writeln('<param name="gbp_103" value="GBPPKR|idc_lite;GBP/PKR;Valuta;0" />');
		document.writeln('<param name="gbp_104" value="GBPPLN|idc_lite;GBP/PLN;Valuta;0" />');
		document.writeln('<param name="gbp_105" value="GBPPYG|idc_lite;GBP/PYG;Valuta;0" />');
		document.writeln('<param name="gbp_106" value="GBPQAR|idc_lite;GBP/QAR;Valuta;0" />');
		document.writeln('<param name="gbp_107" value="GBPRON|idc_lite;GBP/RON;Valuta;0" />');
		document.writeln('<param name="gbp_108" value="GBPRSD|idc_lite;GBP/RSD;Valuta;0" />');
		document.writeln('<param name="gbp_109" value="GBPRUB|idc_lite;GBP/RUB;Valuta;0" />');
		document.writeln('<param name="gbp_110" value="GBPRWF|idc_lite;GBP/RWF;Valuta;0" />');
		document.writeln('<param name="gbp_111" value="GBPSAR|idc_lite;GBP/SAR;Valuta;0" />');
		document.writeln('<param name="gbp_112" value="GBPSCR|idc_lite;GBP/SCR;Valuta;0" />');
		document.writeln('<param name="gbp_113" value="GBPSDG|idc_lite;GBP/SDG;Valuta;0" />');
		document.writeln('<param name="gbp_114" value="GBPSEK|idc_lite;GBP/SEK;Valuta;0" />');
		document.writeln('<param name="gbp_115" value="GBPSGD|idc_lite;GBP/SGD;Valuta;0" />');
		document.writeln('<param name="gbp_116" value="GBPSHP|idc_lite;GBP/SHP;Valuta;0" />');	
		document.writeln('<param name="gbp_117" value="GBPSLL|idc_lite;GBP/SLL;Valuta;0" />');
		document.writeln('<param name="gbp_118" value="GBPSOS|idc_lite;GBP/SOS;Valuta;0" />');
		document.writeln('<param name="gbp_119" value="GBPSRD|idc_lite;GBP/SRD;Valuta;0" />');
		document.writeln('<param name="gbp_120" value="GBPSTD|idc_lite;GBP/STD;Valuta;0" />');
		document.writeln('<param name="gbp_121" value="GBPSVC|idc_lite;GBP/SVC;Valuta;0" />');
		document.writeln('<param name="gbp_122" value="GBPSYP|idc_lite;GBP/SYP;Valuta;0" />');
		document.writeln('<param name="gbp_123" value="GBPSZL|idc_lite;GBP/SZL;Valuta;0" />');
		document.writeln('<param name="gbp_124" value="GBPTHB|idc_lite;GBP/THB;Valuta;0" />');
		document.writeln('<param name="gbp_125" value="GBPTJS|idc_lite;GBP/TJS;Valuta;0" />');
		document.writeln('<param name="gbp_126" value="GBPTMT|idc_lite;GBP/TMT;Valuta;0" />');
		document.writeln('<param name="gbp_127" value="GBPTND|idc_lite;GBP/TND;Valuta;0" />');
		document.writeln('<param name="gbp_128" value="GBPTOP|idc_lite;GBP/TOP;Valuta;0" />');
		document.writeln('<param name="gbp_129" value="GBPTRY|idc_lite;GBP/TRY;Valuta;0" />');
		document.writeln('<param name="gbp_130" value="GBPTTD|idc_lite;GBP/TTD;Valuta;0" />');	
		document.writeln('<param name="gbp_131" value="GBPTWD|idc_lite;GBP/TWD;Valuta;0" />');
		document.writeln('<param name="gbp_132" value="GBPTZS|idc_lite;GBP/TZS;Valuta;0" />');
		document.writeln('<param name="gbp_133" value="GBPUAH|idc_lite;GBP/UAH;Valuta;0" />');
		document.writeln('<param name="gbp_134" value="GBPUGX|idc_lite;GBP/UGX;Valuta;0" />');
		document.writeln('<param name="gbp_135" value="GBPUSD|idc_lite;GBP/USD;Valuta;0" />');
		document.writeln('<param name="gbp_136" value="GBPUYU|idc_lite;GBP/UYU;Valuta;0" />');
		document.writeln('<param name="gbp_137" value="GBPVEF|idc_lite;GBP/VEF;Valuta;0" />');
		document.writeln('<param name="gbp_138" value="GBPVND|idc_lite;GBP/VND;Valuta;0" />');
		document.writeln('<param name="gbp_139" value="GBPVUV|idc_lite;GBP/VUV;Valuta;0" />');
		document.writeln('<param name="gbp_140" value="GBPWST|idc_lite;GBP/WST;Valuta;0" />');
		document.writeln('<param name="gbp_141" value="GBPXCD|idc_lite;GBP/XCD;Valuta;0" />');
		document.writeln('<param name="gbp_142" value="GBPXCU|idc_lite;GBP/XCU;Valuta;0" />');
		document.writeln('<param name="gbp_143" value="GBPXDR|idc_lite;GBP/XDR;Valuta;0" />');
		document.writeln('<param name="gbp_144" value="GBPYER|idc_lite;GBP/YER;Valuta;0" />');
		document.writeln('<param name="gbp_145" value="GBPZAR|idc_lite;GBP/ZAR;Valuta;0" />');
		document.writeln('<param name="gbp_146" value="GBPZMW|idc_lite;GBP/ZMW;Valuta;0" />');
		document.writeln('<param name="gbp_147" value="GBPZWL|idc_lite;GBP/ZWL;Valuta;0" />');			
		document.writeln('<param name="iexplore_market_214_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_215" value="@HTML|usd_|United States Dollar|false|true|false|false" />');
		document.writeln('<param name="usd_1" value="USDAED|idc_lite;USD/AED;Valuta;0" />');
		document.writeln('<param name="usd_2" value="USDAFN|idc_lite;USD/AFN;Valuta;0" />');
		document.writeln('<param name="usd_3" value="USDANG|idc_lite;USD/ANG;Valuta;0" />');
		document.writeln('<param name="usd_4" value="USDARS|idc_lite;USD/ARS;Valuta;0" />');
		document.writeln('<param name="usd_5" value="USDAUD|idc_lite;USD/AUD;Valuta;0" />');
		document.writeln('<param name="usd_6" value="USDAWG|idc_lite;USD/AWG;Valuta;0" />');
		document.writeln('<param name="usd_7" value="USDAZN|idc_lite;USD/AZN;Valuta;0" />');
		document.writeln('<param name="usd_8" value="USDBAM|idc_lite;USD/BAM;Valuta;0" />');
		document.writeln('<param name="usd_9" value="USDBBD|idc_lite;USD/BBD;Valuta;0" />');
		document.writeln('<param name="usd_10" value="USDBDT|idc_lite;USD/BDT;Valuta;0" />');
		document.writeln('<param name="usd_11" value="USDBGN|idc_lite;USD/BGN;Valuta;0" />');
		document.writeln('<param name="usd_12" value="USDBHD|idc_lite;USD/BHD;Valuta;0" />');
		document.writeln('<param name="usd_13" value="USDBIF|idc_lite;USD/BIF;Valuta;0" />');
		document.writeln('<param name="usd_14" value="USDBMD|idc_lite;USD/BMD;Valuta;0" />');
		document.writeln('<param name="usd_15" value="USDBND|idc_lite;USD/BND;Valuta;0" />');
		document.writeln('<param name="usd_16" value="USDBOB|idc_lite;USD/BOB;Valuta;0" />');
		document.writeln('<param name="usd_17" value="USDBRL|idc_lite;USD/BRL;Valuta;0" />');
		document.writeln('<param name="usd_18" value="USDBSD|idc_lite;USD/BSD;Valuta;0" />');
		document.writeln('<param name="usd_19" value="USDBWP|idc_lite;USD/BWP;Valuta;0" />');
		document.writeln('<param name="usd_20" value="USDBYR|idc_lite;USD/BYR;Valuta;0" />');
		document.writeln('<param name="usd_21" value="USDBZD|idc_lite;USD/BZD;Valuta;0" />');
		document.writeln('<param name="usd_22" value="USDCAD|idc_lite;USD/CAD;Valuta;0" />');
		document.writeln('<param name="usd_23" value="USDCDF|idc_lite;USD/CDF;Valuta;0" />');
		document.writeln('<param name="usd_24" value="USDCHF|idc_lite;USD/CHF;Valuta;0" />');
		document.writeln('<param name="usd_25" value="USDCLF|idc_lite;USD/CLF;Valuta;0" />');
		document.writeln('<param name="usd_26" value="USDCLP|idc_lite;USD/CLP;Valuta;0" />');
		document.writeln('<param name="usd_27" value="USDCNH|idc_lite;USD/CNH;Valuta;0" />');
		document.writeln('<param name="usd_28" value="USDCNY|idc_lite;USD/CNY;Valuta;0" />');
		document.writeln('<param name="usd_29" value="USDCOP|idc_lite;USD/COP;Valuta;0" />');
		document.writeln('<param name="usd_30" value="USDCRC|idc_lite;USD/CRC;Valuta;0" />');
		document.writeln('<param name="usd_31" value="USDCUC|idc_lite;USD/CUC;Valuta;0" />');	
		document.writeln('<param name="usd_32" value="USDCVE|idc_lite;USD/CVE;Valuta;0" />');
		document.writeln('<param name="usd_33" value="USDCZK|idc_lite;USD/CZK;Valuta;0" />');
		document.writeln('<param name="usd_34" value="USDDJF|idc_lite;USD/DJF;Valuta;0" />');
		document.writeln('<param name="usd_35" value="USDDKK|idc_lite;USD/DKK;Valuta;0" />');
		document.writeln('<param name="usd_36" value="USDDOP|idc_lite;USD/DOP;Valuta;0" />');
		document.writeln('<param name="usd_37" value="USDDZD|idc_lite;USD/DZD;Valuta;0" />');
		document.writeln('<param name="usd_38" value="USDEGP|idc_lite;USD/EGP;Valuta;0" />');
		document.writeln('<param name="usd_39" value="USDERN|idc_lite;USD/ERN;Valuta;0" />');
		document.writeln('<param name="usd_40" value="USDETB|idc_lite;USD/ETB;Valuta;0" />');
		document.writeln('<param name="usd_41" value="USDEUR|idc_lite;USD/EUR;Valuta;0" />');
		document.writeln('<param name="usd_42" value="USDFKP|idc_lite;USD/FKP;Valuta;0" />');
		document.writeln('<param name="usd_43" value="USDGBP|idc_lite;USD/GBP;Valuta;0" />');
		document.writeln('<param name="usd_44" value="USDGHS|idc_lite;USD/GHS;Valuta;0" />');
		document.writeln('<param name="usd_45" value="USDGIP|idc_lite;USD/GIP;Valuta;0" />');
		document.writeln('<param name="usd_46" value="USDGMD|idc_lite;USD/GMD;Valuta;0" />');
		document.writeln('<param name="usd_47" value="USDGNF|idc_lite;USD/GNF;Valuta;0" />');
		document.writeln('<param name="usd_48" value="USDGTQ|idc_lite;USD/GTQ;Valuta;0" />');
		document.writeln('<param name="usd_49" value="USDGYD|idc_lite;USD/GYD;Valuta;0" />');
		document.writeln('<param name="usd_50" value="USDHKD|idc_lite;USD/HKD;Valuta;0" />');
		document.writeln('<param name="usd_51" value="USDHNL|idc_lite;USD/HNL;Valuta;0" />');
		document.writeln('<param name="usd_52" value="USDHRK|idc_lite;USD/HRK;Valuta;0" />');
		document.writeln('<param name="usd_53" value="USDHTG|idc_lite;USD/HTG;Valuta;0" />');
		document.writeln('<param name="usd_54" value="USDHUF|idc_lite;USD/HUF;Valuta;0" />');
		document.writeln('<param name="usd_55" value="USDIDR|idc_lite;USD/IDR;Valuta;0" />');
		document.writeln('<param name="usd_56" value="USDILS|idc_lite;USD/ILS;Valuta;0" />');
		document.writeln('<param name="usd_57" value="USDINR|idc_lite;USD/INR;Valuta;0" />');
		document.writeln('<param name="usd_58" value="USDIQD|idc_lite;USD/IQD;Valuta;0" />');
		document.writeln('<param name="usd_59" value="USDIRR|idc_lite;USD/IRR;Valuta;0" />');
		document.writeln('<param name="usd_60" value="USDJMD|idc_lite;USD/JMD;Valuta;0" />');
		document.writeln('<param name="usd_61" value="USDJOD|idc_lite;USD/JOD;Valuta;0" />');
		document.writeln('<param name="usd_62" value="USDJPY|idc_lite;USD/JPY;Valuta;0" />');
		document.writeln('<param name="usd_63" value="USDKES|idc_lite;USD/KES;Valuta;0" />');
		document.writeln('<param name="usd_64" value="USDKGS|idc_lite;USD/KGS;Valuta;0" />');
		document.writeln('<param name="usd_65" value="USDKHR|idc_lite;USD/KHR;Valuta;0" />');
		document.writeln('<param name="usd_66" value="USDKMF|idc_lite;USD/KMF;Valuta;0" />');
		document.writeln('<param name="usd_67" value="USDKPW|idc_lite;USD/KPW;Valuta;0" />');
		document.writeln('<param name="usd_68" value="USDKRW|idc_lite;USD/KRW;Valuta;0" />');
		document.writeln('<param name="usd_69" value="USDKWD|idc_lite;USD/KWD;Valuta;0" />');
		document.writeln('<param name="usd_70" value="USDKYD|idc_lite;USD/KYD;Valuta;0" />');
		document.writeln('<param name="usd_71" value="USDKZT|idc_lite;USD/KZT;Valuta;0" />');
		document.writeln('<param name="usd_72" value="USDLAK|idc_lite;USD/LAK;Valuta;0" />');
		document.writeln('<param name="usd_73" value="USDLBP|idc_lite;USD/LBP;Valuta;0" />');
		document.writeln('<param name="usd_74" value="USDLKR|idc_lite;USD/LKR;Valuta;0" />');
		document.writeln('<param name="usd_75" value="USDLRD|idc_lite;USD/LRD;Valuta;0" />');
		document.writeln('<param name="usd_76" value="USDLSL|idc_lite;USD/LSL;Valuta;0" />');
		document.writeln('<param name="usd_77" value="USDLTL|idc_lite;USD/LTL;Valuta;0" />');		
		document.writeln('<param name="usd_78" value="USDLYD|idc_lite;USD/LYD;Valuta;0" />');
		document.writeln('<param name="usd_79" value="USDMAD|idc_lite;USD/MAD;Valuta;0" />');
		document.writeln('<param name="usd_80" value="USDMDL|idc_lite;USD/MDL;Valuta;0" />');
		document.writeln('<param name="usd_81" value="USDMGA|idc_lite;USD/MGA;Valuta;0" />');
		document.writeln('<param name="usd_82" value="USDMKD|idc_lite;USD/MKD;Valuta;0" />');
		document.writeln('<param name="usd_83" value="USDMMK|idc_lite;USD/MMK;Valuta;0" />');
		document.writeln('<param name="usd_84" value="USDMNT|idc_lite;USD/MNT;Valuta;0" />');
		document.writeln('<param name="usd_85" value="USDMOP|idc_lite;USD/MOP;Valuta;0" />');
		document.writeln('<param name="usd_86" value="USDMRO|idc_lite;USD/MRO;Valuta;0" />');
		document.writeln('<param name="usd_87" value="USDMUR|idc_lite;USD/MUR;Valuta;0" />');
		document.writeln('<param name="usd_88" value="USDMVR|idc_lite;USD/MVR;Valuta;0" />');
		document.writeln('<param name="usd_89" value="USDMWK|idc_lite;USD/MWK;Valuta;0" />');
		document.writeln('<param name="usd_90" value="USDMXN|idc_lite;USD/MXN;Valuta;0" />');
		document.writeln('<param name="usd_91" value="USDMYR|idc_lite;USD/MYR;Valuta;0" />');
		document.writeln('<param name="usd_92" value="USDMZN|idc_lite;USD/MZN;Valuta;0" />');
		document.writeln('<param name="usd_93" value="USDNAD|idc_lite;USD/NAD;Valuta;0" />');
		document.writeln('<param name="usd_94" value="USDNGN|idc_lite;USD/NGN;Valuta;0" />');
		document.writeln('<param name="usd_95" value="USDNIO|idc_lite;USD/NIO;Valuta;0" />');
		document.writeln('<param name="usd_96" value="USDNOK|idc_lite;USD/NOK;Valuta;0" />');
		document.writeln('<param name="usd_97" value="USDNPR|idc_lite;USD/NPR;Valuta;0" />');
		document.writeln('<param name="usd_98" value="USDNZD|idc_lite;USD/NZD;Valuta;0" />');
		document.writeln('<param name="usd_99" value="USDOMR|idc_lite;USD/OMR;Valuta;0" />');
		document.writeln('<param name="usd_100" value="USDPAB|idc_lite;USD/PAB;Valuta;0" />');
		document.writeln('<param name="usd_101" value="USDPEN|idc_lite;USD/PEN;Valuta;0" />');
		document.writeln('<param name="usd_102" value="USDPGK|idc_lite;USD/PGK;Valuta;0" />');
		document.writeln('<param name="usd_103" value="USDPHP|idc_lite;USD/PHP;Valuta;0" />');
		document.writeln('<param name="usd_104" value="USDPKR|idc_lite;USD/PKR;Valuta;0" />');
		document.writeln('<param name="usd_105" value="USDPLN|idc_lite;USD/PLN;Valuta;0" />');
		document.writeln('<param name="usd_106" value="USDPYG|idc_lite;USD/PYG;Valuta;0" />');
		document.writeln('<param name="usd_107" value="USDQAR|idc_lite;USD/QAR;Valuta;0" />');
		document.writeln('<param name="usd_108" value="USDRON|idc_lite;USD/RON;Valuta;0" />');
		document.writeln('<param name="usd_109" value="USDRUB|idc_lite;USD/RUB;Valuta;0" />');		
		document.writeln('<param name="usd_110" value="USDRWF|idc_lite;USD/RWF;Valuta;0" />');
		document.writeln('<param name="usd_111" value="USDSAR|idc_lite;USD/SAR;Valuta;0" />');
		document.writeln('<param name="usd_112" value="USDSDG|idc_lite;USD/SDG;Valuta;0" />');
		document.writeln('<param name="usd_113" value="USDSEK|idc_lite;USD/SEK;Valuta;0" />');
		document.writeln('<param name="usd_114" value="USDSGD|idc_lite;USD/SGD;Valuta;0" />');
		document.writeln('<param name="usd_115" value="USDSHP|idc_lite;USD/SHP;Valuta;0" />');
		document.writeln('<param name="usd_116" value="USDSLL|idc_lite;USD/SLL;Valuta;0" />');
		document.writeln('<param name="usd_117" value="USDSOS|idc_lite;USD/SOS;Valuta;0" />');
		document.writeln('<param name="usd_118" value="USDSRD|idc_lite;USD/SRD;Valuta;0" />');
		document.writeln('<param name="usd_119" value="USDSTD|idc_lite;USD/STD;Valuta;0" />');
		document.writeln('<param name="usd_120" value="USDSVC|idc_lite;USD/SVC;Valuta;0" />');
		document.writeln('<param name="usd_121" value="USDSYP|idc_lite;USD/SYP;Valuta;0" />');
		document.writeln('<param name="usd_122" value="USDSZL|idc_lite;USD/SZL;Valuta;0" />');
		document.writeln('<param name="usd_123" value="USDTHB|idc_lite;USD/THB;Valuta;0" />');
		document.writeln('<param name="usd_124" value="USDTJS|idc_lite;USD/TJS;Valuta;0" />');
		document.writeln('<param name="usd_125" value="USDTMT|idc_lite;USD/TMT;Valuta;0" />');
		document.writeln('<param name="usd_126" value="USDTND|idc_lite;USD/TND;Valuta;0" />');
		document.writeln('<param name="usd_127" value="USDTOP|idc_lite;USD/TOP;Valuta;0" />');
		document.writeln('<param name="usd_128" value="USDTRY|idc_lite;USD/TRY;Valuta;0" />');
		document.writeln('<param name="usd_128" value="USDTTD|idc_lite;USD/TTD;Valuta;0" />');
		document.writeln('<param name="usd_130" value="USDTWD|idc_lite;USD/TWD;Valuta;0" />');
		document.writeln('<param name="usd_131" value="USDTZS|idc_lite;USD/TZS;Valuta;0" />');
		document.writeln('<param name="usd_132" value="USDUAH|idc_lite;USD/UAH;Valuta;0" />');
		document.writeln('<param name="usd_133" value="USDUGX|idc_lite;USD/UGX;Valuta;0" />');
		document.writeln('<param name="usd_134" value="USDUSD|idc_lite;USD/USD;Valuta;0" />');
		document.writeln('<param name="usd_135" value="USDUYU|idc_lite;USD/UYU;Valuta;0" />');
		document.writeln('<param name="usd_136" value="USDVEF|idc_lite;USD/VEF;Valuta;0" />');
		document.writeln('<param name="usd_137" value="USDVND|idc_lite;USD/VND;Valuta;0" />');
		document.writeln('<param name="usd_138" value="USDVUV|idc_lite;USD/VUV;Valuta;0" />');
		document.writeln('<param name="usd_139" value="USDWST|idc_lite;USD/WST;Valuta;0" />');
		document.writeln('<param name="usd_140" value="USDXAF|idc_lite;USD/XAF;Valuta;0" />');
		document.writeln('<param name="usd_141" value="USDXCD|idc_lite;USD/XCD;Valuta;0" />');
		document.writeln('<param name="usd_142" value="USDXCU|idc_lite;USD/XCU;Valuta;0" />');
		document.writeln('<param name="usd_143" value="USDXOF|idc_lite;USD/XOF;Valuta;0" />');
		document.writeln('<param name="usd_144" value="USDXPF|idc_lite;USD/YER;Valuta;0" />');
		document.writeln('<param name="usd_145" value="USDZAR|idc_lite;USD/ZAR;Valuta;0" />');
		document.writeln('<param name="usd_146" value="USDZMW|idc_lite;USD/ZMW;Valuta;0" />');
		document.writeln('<param name="usd_147" value="USDZWL|idc_lite;USD/ZWL;Valuta;0" />');
		document.writeln('<param name="iexplore_market_215_path" value="Cross Rates" />');

		document.writeln('<param name="iexplore_market_216" value="@HTML|vef_|Venezuelan Bolivar Fuerte|false|true|false|false" />');
		document.writeln('<param name="vef_1" value="VEFBRL|idc_lite;VEF/BRL;Valuta;0" />');	
		document.writeln('<param name="vef_2" value="VEFGBP|idc_lite;VEF/GBP;Valuta;0" />');
		document.writeln('<param name="vef_3" value="AUDVEF|idc_lite;AUD/VEF;Valuta;0" />');	
		document.writeln('<param name="vef_4" value="BRLVEF|idc_lite;BRL/VEF;Valuta;0" />');	
		document.writeln('<param name="vef_5" value="CADVEF|idc_lite;CAD/VEF;Valuta;0" />');	
		document.writeln('<param name="vef_6" value="CHFVEF|idc_lite;CHF/VEF;Valuta;0" />');	
		document.writeln('<param name="vef_7" value="EURVEF|idc_lite;EUR/VEF;Valuta;0" />');	
		document.writeln('<param name="vef_8" value="GBPVEF|idc_lite;GBP/VEF;Valuta;0" />');	
		document.writeln('<param name="vef_9" value="JPYVEF|idc_lite;JPY/VEF;Valuta;0" />');	
		document.writeln('<param name="vef_10" value="USDVEF|idc_lite;USD/VEF;Valuta;0" />');	
		document.writeln('<param name="iexplore_market_216_path" value="Cross Rates" />');
		
		document.writeln('<param name="iexplore_market_217" value="@HTML|vnd_|Vietnamese Dong|false|true|false|false" />');
		document.writeln('<param name="vnd_1" value="VNDAED|idc_lite;VND/AED;Valuta;0" />');	
		document.writeln('<param name="vnd_2" value="VNDAUD|idc_lite;VND/AUD;Valuta;0" />');		
		document.writeln('<param name="vnd_3" value="VNDCAD|idc_lite;VND/CAD;Valuta;0" />');	
		document.writeln('<param name="vnd_4" value="VNDEUR|idc_lite;VND/EUR;Valuta;0" />');	
		document.writeln('<param name="vnd_5" value="VNDGBP|idc_lite;VND/GBP;Valuta;0" />');	
		document.writeln('<param name="vnd_6" value="VNDHKD|idc_lite;VND/HKD;Valuta;0" />');	
		document.writeln('<param name="vnd_7" value="VNDINR|idc_lite;VND/INR;Valuta;0" />');	
		document.writeln('<param name="vnd_8" value="VNDJPY|idc_lite;VND/JPY;Valuta;0" />');		
		document.writeln('<param name="vnd_9" value="VNDMYR|idc_lite;VND/MYR;Valuta;0" />');	
		document.writeln('<param name="vnd_10" value="VNDPHP|idc_lite;VND/PHP;Valuta;0" />');	
		document.writeln('<param name="vnd_11" value="VNDSGD|idc_lite;VND/SGD;Valuta;0" />');	
		document.writeln('<param name="vnd_12" value="VNDTHB|idc_lite;VND/THB;Valuta;0" />');
		document.writeln('<param name="vnd_13" value="VNDTWD|idc_lite;VND/TWD;Valuta;0" />');	
		document.writeln('<param name="iexplore_market_217_path" value="Cross Rates" />');
		
        	document.writeln('<param name="iexplore_market_218" value="@HTML|usdx_|US Dollar index|false|true|false|false" />');
		document.writeln('<param name="usdx_1" value="!DXX|idc_dla;US Dollar Index;Valuta;0" />');
		
		document.writeln('<param name="iexplore_market_219" value="@HTML|xan_|Precious Metals - NetDania|false|true|false|false" />');
		
        document.writeln('<param name="xan_1" value="XAUAUD|netdania_fxa;Gold, spot (AUD);Valuta;0" />');
        document.writeln('<param name="xan_2" value="XAUCAD|netdania_fxa;Gold, spot (CAD);Valuta;0" />');
        document.writeln('<param name="xan_3" value="XAUCHF|netdania_fxa;Gold, spot (CHF);Valuta;0" />');
        document.writeln('<param name="xan_4" value="XAUEUR|netdania_fxa;Gold, spot (EUR);Valuta;0" />');
        document.writeln('<param name="xan_5" value="XAUGBP|netdania_fxa;Gold, spot (GBP);Valuta;0" />');
        document.writeln('<param name="xan_6" value="XAUJPY|netdania_fxa;Gold, spot (JPY);Valuta;0" />');
        document.writeln('<param name="xan_7" value="XAUUSD|netdania_rt;Gold, spot (USD);Valuta;0" />');
        document.writeln('<param name="xan_8" value="XAUTRY|netdania_rt;Gold, spot (TRY);Valuta;0" />');
        document.writeln('<param name="xan_9" value="XAUNZD|netdania_fxa;Gold, spot (NZD);Valuta;0" />');
        document.writeln('<param name="xan_10" value="XAUSGD|netdania_fxa;Gold, spot (SGD);Valuta;0" />');      
        document.writeln('<param name="xan_11" value="XAGAUD|netdania_fxa;Silver, spot (AUD);Valuta;0" />');
        document.writeln('<param name="xan_12" value="XAGCAD|netdania_fxa;Silver, spot (CAD);Valuta;0" />');
        document.writeln('<param name="xan_13" value="XAGCHF|netdania_fxa;Silver, spot (CHF);Valuta;0" />');
        document.writeln('<param name="xan_14" value="XAGEUR|netdania_fxa;Silver, spot (EUR);Valuta;0" />');
        document.writeln('<param name="xan_15" value="XAGGBP|netdania_fxa;Silver, spot (GBP);Valuta;0" />');
        document.writeln('<param name="xan_16" value="XAGJPY|netdania_fxa;Silver, spot (JPY);Valuta;0" />');
        document.writeln('<param name="xan_17" value="XAGUSD|netdania_rt;Silver, spot (USD);Valuta;0" />');       
        document.writeln('<param name="xan_18" value="XAUXAG|netdania_fxa;Gold/Silver ratio;Valuta;0" />');
				 
		document.writeln('<param name="iexplore_market_220" value="@HTML|xa_|Precious Metals - IDC|false|true|false|false" />');
		
		document.writeln('<param name="xa_1" value="XAUAUDOZ|idc_lite;Gold, spot (AUD);Valuta;0" />');
		document.writeln('<param name="xa_2" value="XAUCADOZ|idc_lite;Gold, spot (CAD);Valuta;0" />');
		document.writeln('<param name="xa_3" value="XAUCHFOZ|idc_lite;Gold, spot (CHF);Valuta;0" />');
		document.writeln('<param name="xa_4" value="XAUEUROZ|idc_lite;Gold, spot (EUR);Valuta;0" />');
		document.writeln('<param name="xa_5" value="XAUGBPOZ|idc_lite;Gold, spot (GBP);Valuta;0" />');
		document.writeln('<param name="xa_6" value="XAUJPYOZ|idc_lite;Gold, spot (JPY);Valuta;0" />');
		document.writeln('<param name="xa_7" value="XAUUSDOZ|idc_lite;Gold, spot (USD);Valuta;0" />');
	
		document.writeln('<param name="xa_8" value="XAGAUDOZ|idc_lite;Silver, spot (AUD);Valuta;0" />');
		document.writeln('<param name="xa_9" value="XAGCADOZ|idc_lite;Silver, spot (CAD);Valuta;0" />');
		document.writeln('<param name="xa_10" value="XAGCHFOZ|idc_lite;Silver, spot (CHF);Valuta;0" />');
		document.writeln('<param name="xa_11" value="XAGEUROZ|idc_lite;Silver, spot (EUR);Valuta;0" />');
		document.writeln('<param name="xa_12" value="XAGGBPOZ|idc_lite;Silver, spot (GBP);Valuta;0" />');
		document.writeln('<param name="xa_13" value="XAGUSDOZ|idc_lite;Silver, spot (USD);Valuta;0" />');
		document.writeln('<param name="xa_14" value="XPDUSDOZ|idc_lite;Palladium;Valuta;0" />');
		document.writeln('<param name="xa_15" value="XPTUSDOZ|idc_lite;Platinum;Valuta;0" />');

		document.writeln('<param name="iexplore_market_221" value="@HTML|oil_|ICE Futures|false|true|false|false" />');
		document.writeln('<param name="oil_1" value="EB0Y.ip|idc_dla;Brent Crude Fut. 1st. pos.;Valuta;0" />');
		document.writeln('<param name="oil_2" value="EP0Y.ip|idc_dla;Gas Oil, 1. pos.;Valuta;0" />');
		document.writeln('<param name="oil_3" value="NG0Y.ip|idc_dla;UK Natural Gas, 1. pos.;Valuta;0" />');
		document.writeln('<param name="oil_4" value="ET0Y.ip|idc_dla;WTI Crude 1. pos.;Valuta;0" />');


		document.writeln('<param name="quotes_instruments_title" value="Instrument Explorer" />');
		document.writeln('<param name="iexplore_loading_icon" value="loading.gif" />');

  		document.writeln('<param name="iexplore_cannot_drop_message" value="Cannot drop this node to specified location!" />');
  		document.writeln('<param name="iexplore_cannot_drop_to_an_instrument_node_message" value="Cannot drop a node to an instrument node!" />');
  		document.writeln('<param name="iexplore_cannot_rename_node_message" value="Cannot rename the specified node!" />');
  		document.writeln('<param name="iexplore_confirm_node_delete_message" value="Confirm Node Delete" />');
  		document.writeln('<param name="iexplore_confirm_node_delete_question" value="Are you sure you want to remove this node?" />');
  		document.writeln('<param name="iexplore_duplicate_node_cannot_paste_message" value="Cannot paste here!" />');
  		document.writeln('<param name="iexplore_duplicate_node_message" value="Duplicate node" />');
  		document.writeln('<param name="iexplore_edit_instrument_dialog_title" value="Edit Instrument" />');
  		document.writeln('<param name="iexplore_edit_instrument_name_label" value="Instrument name: " />');
  		document.writeln('<param name="iexplore_edit_instrument_symbol_label" value="Instrument symbol: " />');
  		document.writeln('<param name="iexplore_error_message" value="Error" />');
  		document.writeln('<param name="iexplore_favorites_name" value="Favorites" />');
  		document.writeln('<param name="iexplore_group_all_instruments_label" value="All" />');
  		document.writeln('<param name="iexplore_group_by_market_label" value="Market" />');
  		document.writeln('<param name="iexplore_group_by_type_label" value="Type" />');
  		document.writeln('<param name="iexplore_group_instruments_tooltip_message" value="Group instruments" />');
  		document.writeln('<param name="iexplore_instrument_type_all_types" value="All Types" />');
  		document.writeln('<param name="iexplore_instrument_type_bond" value="Bond" />');
  		document.writeln('<param name="iexplore_instrument_type_etched_label_name" value="Instrument type:" />');
  		document.writeln('<param name="iexplore_instrument_type_futures" value="Futures" />');
  		document.writeln('<param name="iexplore_instrument_type_index" value="Indices" />');
  		document.writeln('<param name="iexplore_instrument_type_not_classified" value="Not Classified" />');
  		document.writeln('<param name="iexplore_instrument_type_option" value="Option" />');
  		document.writeln('<param name="iexplore_instrument_type_stock" value="Shares" />');
  		document.writeln('<param name="iexplore_instrument_type_warrant" value="Warrant" />');
  		document.writeln('<param name="iexplore_instruments_tab_name" value="Instruments" />');
  		document.writeln('<param name="iexplore_last_used_name" value="Recent" />');
  		document.writeln('<param name="iexplore_loading_message" value="Loading instruments..." />');
  		document.writeln('<param name="iexplore_market_etched_label_name" value="Market:" />');
  		document.writeln('<param name="iexplore_new_category_default_name" value="New Category" />');
  		document.writeln('<param name="iexplore_new_instrument_dialog_title" value="New Instrument" />');
  		document.writeln('<param name="iexplore_no_instruments_found_message" value="No instruments found!" />');
  		document.writeln('<param name="iexplore_no_instruments_message" value="No instruments available." />');
  		document.writeln('<param name="iexplore_popupmenu_collapse_item_label" value="Collapse" />');
  		document.writeln('<param name="iexplore_popupmenu_collapse_item_tooltip" value="Collapse this category." />');
  		document.writeln('<param name="iexplore_popupmenu_copy_item_label" value="Copy" />');
  		document.writeln('<param name="iexplore_popupmenu_copy_item_tooltip" value="Copy this node to clipboard." />');
  		document.writeln('<param name="iexplore_popupmenu_cut_item_label" value="Cut" />');
  		document.writeln('<param name="iexplore_popupmenu_cut_item_tooltip" value="Cut this node to clipboard." />');
  		document.writeln('<param name="iexplore_popupmenu_delete_item_label" value="Delete" />');
  		document.writeln('<param name="iexplore_popupmenu_delete_item_tooltip" value="Delete this node." />');
  		document.writeln('<param name="iexplore_popupmenu_edit_item_label" value="Edit" />');
  		document.writeln('<param name="iexplore_popupmenu_edit_item_tooltip" value="Edit this node." />');
  		document.writeln('<param name="iexplore_popupmenu_expand_item_label" value="Expand" />');
  		document.writeln('<param name="iexplore_popupmenu_expand_item_tooltip" value="Expand this category." />');
  		document.writeln('<param name="iexplore_popupmenu_move_down_item_label" value="Move Down" />');
  		document.writeln('<param name="iexplore_popupmenu_move_down_item_tooltip" value="Move down this node." />');
  		document.writeln('<param name="iexplore_popupmenu_move_up_item_label" value="Move Up" />');
  		document.writeln('<param name="iexplore_popupmenu_move_up_item_tooltip" value="Move up this node." />');
  		document.writeln('<param name="iexplore_popupmenu_new_category_item_label" value="New Category" />');
  		document.writeln('<param name="iexplore_popupmenu_new_category_item_tooltip" value="Add a category node here." />');
  		document.writeln('<param name="iexplore_popupmenu_new_instrument_item_label" value="New Instrument" />');
  		document.writeln('<param name="iexplore_popupmenu_new_instrument_item_tooltip" value="Add an instrument node here." />');
  		document.writeln('<param name="iexplore_popupmenu_new_item_label" value="New" />');
  		document.writeln('<param name="iexplore_popupmenu_paste_item_label" value="Paste" />');
  		document.writeln('<param name="iexplore_popupmenu_paste_item_tooltip" value="Paste here from clipboard." />');
  		document.writeln('<param name="iexplore_popupmenu_recent_item_label" value="Recent" />');
  		document.writeln('<param name="iexplore_popupmenu_rename_item_label" value="Rename" />');
  		document.writeln('<param name="iexplore_popupmenu_rename_item_tooltip" value="Rename this node." />');
  		document.writeln('<param name="iexplore_popupmenu_sendto_item_label" value="Send To" />');
  		document.writeln('<param name="iexplore_results_name" value="Search Result" />');
  		document.writeln('<param name="iexplore_search_for_label_name" value="Search for:" />');
  		document.writeln('<param name="iexplore_search_for_name_button_label" value="Name" />');
  		document.writeln('<param name="iexplore_search_for_symbol_button_label" value="Symbol" />');
  		document.writeln('<param name="iexplore_search_label" value="Search for" />');
  		document.writeln('<param name="iexplore_search_mode_contains_label_name" value="Contains" />');
  		document.writeln('<param name="iexplore_search_mode_ends_with_label_name" value="Ends with" />');
  		document.writeln('<param name="iexplore_search_mode_etched_label_name" value="Search mode:" />');
  		document.writeln('<param name="iexplore_search_mode_starts_with_label_name" value="Starts with" />');
 		document.writeln('<param name="iexplore_options_tab_name" value="Options" />');
		document.writeln('<param name="iexplore_favorites_in_tree" value="true" />');


		<!-- InstrumentExplorer end -->
}


function loadLookAndFeel()
{
		document.writeln('<param name="LNF_2" value="Metal;-1000;javax.swing.plaf.metal.MetalLookAndFeel"/>');
		
		if(BrowserDetect.OS == "Mac")
		{
		    document.writeln('<param name="default_lnf" value="apple.laf.AquaLookAndFeel"/>');
		    document.writeln('<param name="LNF_1" value="Aqua;-1000;apple.laf.AquaLookAndFeel"/>');
		}
		else
		{
		    document.writeln('<param name="LNF_1" value="Windows;-1000;com.sun.java.swing.plaf.windows.WindowsLookAndFeel"/>');
		    document.writeln('<param name="default_lnf" value="com.sun.java.swing.plaf.windows.WindowsLookAndFeel"/>');
		}
}


var install;
function writeMessageMultipleJVMs()
{
	
	document.writeln(
						'\
					<div id="javaDetectMessage"><h3><span align="center" style="color: #648cc7;font-weight: bold;">Multiple Java versions installed</span></h3>\
					<p>It seems you have multiple java versions installed on your PC generating a conflict.\
					In order to fix this problem we suggest following the steps:</p>\
					1. Uninstall all Java versions as described <a style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" target ="blank" href="http://www.netdania.com/UI/DisplayKBItem.aspx?kbiID=9ac2f4cc-a708-461e-a217-64a78fd14e6b&menu=Support">here</a>.<br/><br/>\
					2. Download and install the latest Java version by clicking the button below:<br/>\
					 <a href="http://java.com/java/download/index.jsp?cid=jdp79239" target="_blank">\
                    <img width="88" height="31" border="0" style="margin: 8px;" alt="GetJava Download Button"\
                        title="GetJava" src="http://java.com/en/img/everywhere/getjava_sm.gif?cid=jdp79239"></a>\
					<p>3. Restart your browser.</p>\
					If you encounter further problems please don\'t hesitate to\
					<a style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" target ="blank" href="/UI/JavaHelp.aspx">contact us</a>\
					(use the form at the bottom).<p/>\
					<span onmouseover="mouseon(this)" style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" onmouseout="mouseoff(this)" onclick="closeDetectMessage()">Close this message</span>\
					</div>'
					);
}	

function writeMessageMissingCorrectJVM()
{
	document.writeln(
					'\
					<div id="javaDetectMessage"><h3><span style="color: #648cc7;font-weight: bold;">The required Java version is not installed</span></h3>\
					<p>In order to run this application a newer Java plug-in is required.<p/>\
					To fix this problem we suggest following the steps:</p>\
					1. Uninstall all Java versions as described <a style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" target ="blank" href="http://www.netdania.com/UI/DisplayKBItem.aspx?kbiID=9ac2f4cc-a708-461e-a217-64a78fd14e6b&menu=Support">here</a>.<br/><br/>\
					2. Download and install the latest Java version by clicking the button below:<br/>\
					 <a href="http://java.com/java/download/index.jsp?cid=jdp79239" target="_blank">\
                    <img width="88" height="31" border="0" style="margin: 8px;" alt="GetJava Download Button"\
                        title="GetJava" src="http://java.com/en/img/everywhere/getjava_sm.gif?cid=jdp79239"></a>\
					<p>3. Restart your browser.</p>\
					If you encounter further problems please don\'t hesitate to\
					<a style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" target ="blank" href="/UI/JavaHelp.aspx">contact us</a>\
					(use the form at the bottom).<p/>\
					<span onmouseover="mouseon(this)" style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" onmouseout="mouseoff(this)" onclick="closeDetectMessage()">Close this message</span>\
					</div>'
					);
}

function writeMessNoJavaInstalled()
{
	
	
	if(typeof(deployJava) != "undefined" && BrowserDetect.browser != 'Opera')
	{
	document.writeln(
					'\
					<div id="javaDetectMessage"><h3><span style="color: #648cc7;font-weight: bold;">No Java Installed</span></h3>\
					<p>In order to run this application, a Java plug-in has to be installed on your PC.<p/>\
					The latest Java version will be installed automatically. If the installation doesn\'t start in 20 seconds click\
					<span onmouseover="mouseon(this)" style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" onmouseout="mouseoff(this)" onClick="deployJava.installLatestJRE()">here</span>,\
					or you can click the button below to manually download and install it.<br/>\
					 <a href="http://java.com/java/download/index.jsp?cid=jdp79239" target="_blank">\
                    <img width="88" height="31" border="0" style="margin: 8px;" alt="GetJava Download Button"\
                        title="GetJava" src="http://java.com/en/img/everywhere/getjava_sm.gif?cid=jdp79239"></a>\
					<p>Restart your browser to complete the installation.</p>\
					<p>Click <span onmouseover="mouseon(this)" style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" onmouseout="mouseoff(this)" onclick="cancelDownload()">here</span> to cancel automatic installation.</p>\
					If you encounter further problems please don\'t hesitate to\
					<a style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" target ="blank" href="/UI/JavaHelp.aspx">contact us</a>\
					(use the form at the bottom).<p/>\
					<span onmouseover="mouseon(this)" style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" onmouseout="mouseoff(this)" onclick="closeDetectMessage()">Close this message</span>\
					</div>'
					);
					install = setTimeout("deployJava.installLatestJRE()", 20000);
	}
	else
	{
		document.writeln(
					'\
					<div id="javaDetectMessage"><h3><span style="color: #648cc7;font-weight: bold;">No Java Installed</span></h3>\
					<p>In order to run this application, a Java plug-in has to be installed on your PC.<p/>\
					You can click the button below to manually download and install Java.<br/>\
					 <a href="http://java.com/java/download/index.jsp?cid=jdp79239" target="_blank">\
                    <img width="88" height="31" border="0" style="margin: 8px;" alt="GetJava Download Button"\
                        title="GetJava" src="http://java.com/en/img/everywhere/getjava_sm.gif?cid=jdp79239"></a>\
					<p>Restart your browser to complete the installation.</p>\
					If you encounter further problems please don\'t hesitate to\
					<a style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" target ="blank" href="/UI/JavaHelp.aspx">contact us</a>\
					(use the form at the bottom).<p/>\
					<span onmouseover="mouseon(this)" style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" onmouseout="mouseoff(this)" onclick="closeDetectMessage()">Close this message</span>\
					</div>'
					);
					
	}
}	

function closeDetectMessage()
{
	var detectMess = document.getElementById('javaDetectMessage');
	var detectMessDD = document.getElementById('userMessDetectDone');
	if(detectMess != null)
		detectMess.style.display = 'none';
	if(detectMessDD != null)
		detectMessDD.style.display = 'none';
}

function compareVersion(installed,required){
		var a=installed.split('.');
		var b=required.split('.');
		
		for(var i=0;i<a.length;++i){
			a[i]=Number(a[i]);
		}
		for(var i=0;i<b.length;++i){
			b[i]=Number(b[i]);
		}
		
		if(a.length==2){
			a[2]=0;
		}

		if(a[0]>b[0])
			return true;
		
		if(a[0]<b[0])
			return false;
		
		if(a[1]>b[1])
			return true;
		if(a[1]<b[1])
			return false;
		if(a[2]>b[2])
			return true;
		if(a[2]<b[2])
			return false;
		return true;
	}

	function IsCorrectVersionInstalled(list)
	{
		var correctVersionExists = false;
		
		for(var i=0;i<list.length;++i){
			
			if(compareVersion(list[i],requiredJavaVersion))
				return true;
		}
		return correctVersionExists;
	}
	

function cancelDownload()
{
	clearTimeout(install);
	alert("Automatic download is cancelled!");
}

function showMessageDetectJava()
{
	BrowserDetect.init();	
    var requiredJavaVersion = '1.5.0';     
	var runningJavaVersion = document.getElementById("javaversion").value;	
	var JVMs = [];
	
	if(typeof(deployJava) != "undefined")
	{
		JVMs = deployJava.getJREs();
	}
	else
	{
		JVMs = INFO.DeployTK_versions;
	}
	
	if(runningJavaVersion == 'Unknown' || runningJavaVersion == null || runningJavaVersion == "wait")
	{		
		
		if(JVMs.length > 0)
		{
			//MAC
			if(BrowserDetect.OS == 'Mac')
			{
				if(compareVersion(JVMs[0],requiredJavaVersion)==true)
				{
					writeMessageMultipleJVMs();
				}
				else
				{
					writeMessageMissingCorrectJVM();
				}
			}
			else
			{
				if(JVMs.length > 1)
				{
					if(IsCorrectVersionInstalled(JVMs))
					{
						writeMessageMultipleJVMs();
					}
					else
					{
						writeMessageMissingCorrectJVM();
					}
				}
				else
				{
					if(compareVersion(JVMs[0],requiredJavaVersion)==true)
					{
						writeMessageMultipleJVMs();
						//correctVersion = true;
					}
					else
					{
						writeMessageMissingCorrectJVM();
					}
				}
			}
					
		}
		else
		{		
			writeMessNoJavaInstalled();
		}
	}
	else if(compareVersion(runningJavaVersion,requiredJavaVersion)==true)
	{
		correctVersion = true;
	}
	else
	{
		if(BrowserDetect.OS == 'Mac')
		{
			
			if(JVMs.length > 0)
			{
				if(runningJavaVersion != JVMs[0])
				{
					if(compareVersion(JVMs[0],requiredJavaVersion)==true)
					{
						writeMessageMultipleJVMs();
					}
					else
					{
						writeMessageMissingCorrectJVM();
					}
				}
				else
				{
					writeMessageMissingCorrectJVM();
				}
			}
			else
			{	
				writeMessageMissingCorrectJVM();
			}
		}
		else
		{
			if(JVMs.length > 1)
			{
				if(IsCorrectVersionInstalled(JVMs)==true)
				{
					writeMessageMultipleJVMs();
				}
				else
				{
					 writeMessageMissingCorrectJVM();
				}
			}
			else if(runningJavaVersion != JVMs[0]){
					
					if(compareVersion(JVMs[0],requiredJavaVersion)==true)
					{
						writeMessageMultipleJVMs();
					}
					else
					{
						writeMessageMissingCorrectJVM();
					}
			}
			else
			{
				writeMessageMissingCorrectJVM();
			}
		}
	}
	
}


var messNoJavaInstalledNoDeploy = 
					'\
					<div id="javaDetectMessage"><h3><span style="color: #648cc7;font-weight: bold;">No Java Installed</span></h3>\
					<p>In order to run this application, a Java plug-in has to be installed on your PC.<p/>\
					You can click the button below to manually download and install Java.<br/>\
					 <a href="http://java.com/java/download/index.jsp?cid=jdp79239" target="_blank">\
                    <img width="88" height="31" border="0" style="margin: 8px;" alt="GetJava Download Button"\
                        title="GetJava" src="http://java.com/en/img/everywhere/getjava_sm.gif?cid=jdp79239"></a>\
					<p>Restart your browser to complete the installation.</p>\
					If you encounter further problems please don\'t hesitate to\
					<a style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" target ="blank" href="/UI/JavaHelp.aspx">contact us</a>\
					(use the form at the bottom).<p/>\
					<span onmouseover="mouseon(this)" style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" onmouseout="mouseoff(this)" onclick="closeDetectMessage()">Close this message</span>\
					</div>';


var messNoJavaInstalledDeploy = 
'\
					<div id="javaDetectMessage"><h3><span style="color: #648cc7;font-weight: bold;">No Java Installed</span></h3>\
					<p>In order to run this application, a Java plug-in has to be installed on your PC.<p/>\
					The latest Java version will be installed automatically. If the installation doesn\'t start in 20 seconds click\
					<span onmouseover="mouseon(this)" style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" onmouseout="mouseoff(this)" onClick="deployJava.installLatestJRE()">here</span>,\
					or you can click the button below to manually download and install it.<br/>\
					 <a href="http://java.com/java/download/index.jsp?cid=jdp79239" target="_blank">\
                    <img width="88" height="31" border="0" style="margin: 8px;" alt="GetJava Download Button"\
                        title="GetJava" src="http://java.com/en/img/everywhere/getjava_sm.gif?cid=jdp79239"></a>\
					<p>Restart your browser to complete the installation.</p>\
					<p>Click <span onmouseover="mouseon(this)" style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" onmouseout="mouseoff(this)" onclick="cancelDownload()">here</span> to cancel automatic installation.</p>\
					If you encounter further problems please don\'t hesitate to\
					<a style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" target ="blank" href="/UI/JavaHelp.aspx">contact us</a>\
					(use the form at the bottom).<p/>\
					<span onmouseover="mouseon(this)" style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" onmouseout="mouseoff(this)" onclick="closeDetectMessage()">Close this message</span>\
					</div>';

var messageMissingCorrectJVM = 
'\
					<div id="javaDetectMessage"><h3><span style="color: #648cc7;font-weight: bold;">The required Java version is not installed</span></h3>\
					<p>In order to run this application a newer Java plug-in is required.<p/>\
					To fix this problem we suggest following the steps:</p>\
					1. Uninstall all Java versions as described <a style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" target ="blank" href="http://www.netdania.com/UI/DisplayKBItem.aspx?kbiID=9ac2f4cc-a708-461e-a217-64a78fd14e6b&menu=Support">here</a>.<br/><br/>\
					2. Download and install the latest Java version by clicking the button below:<br/>\
					 <a href="http://java.com/java/download/index.jsp?cid=jdp79239" target="_blank">\
                    <img width="88" height="31" border="0" style="margin: 8px;" alt="GetJava Download Button"\
                        title="GetJava" src="http://java.com/en/img/everywhere/getjava_sm.gif?cid=jdp79239"></a>\
					<p>3. Restart your browser.</p>\
					If you encounter further problems please don\'t hesitate to\
					<a style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" target ="blank" href="/UI/JavaHelp.aspx">contact us</a>\
					(use the form at the bottom).<p/>\
					<span onmouseover="mouseon(this)" style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" onmouseout="mouseoff(this)" onclick="closeDetectMessage()">Close this message</span>\
					</div>';

var messageMultipleJVMs = 
'\
					<div id="javaDetectMessage"><h3><span align="center" style="color: #648cc7;font-weight: bold;">Multiple Java versions installed</span></h3>\
					<p>It seems you have multiple java versions installed on your PC generating a conflict.\
					In order to fix this problem we suggest following the steps:</p>\
					1. Uninstall all Java versions as described <a style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" target ="blank" href="http://www.netdania.com/UI/DisplayKBItem.aspx?kbiID=9ac2f4cc-a708-461e-a217-64a78fd14e6b&menu=Support">here</a>.<br/><br/>\
					2. Download and install the latest Java version by clicking the button below:<br/>\
					 <a href="http://java.com/java/download/index.jsp?cid=jdp79239" target="_blank">\
                    <img width="88" height="31" border="0" style="margin: 8px;" alt="GetJava Download Button"\
                        title="GetJava" src="http://java.com/en/img/everywhere/getjava_sm.gif?cid=jdp79239"></a>\
					<p>3. Restart your browser.</p>\
					If you encounter further problems please don\'t hesitate to\
					<a style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" target ="blank" href="/UI/JavaHelp.aspx">contact us</a>\
					(use the form at the bottom).<p/>\
					<span onmouseover="mouseon(this)" style="color: #648cc7;font-style: normal; text-decoration: none; font-weight: bold;" onmouseout="mouseoff(this)" onclick="closeDetectMessage()">Close this message</span>\
					</div>';
					
					
messNoJavaInstalledNoDeploy = "";
messNoJavaInstalledDeploy = "";
messageMissingCorrectJVM = "";
messageMultipleJVMs = "";

function showMessageDetectJavaOnDD()
{
    var requiredJavaVersion = '1.5.0';     
	var runningJavaVersion = document.getElementById("javaversion").value;	
	var JVMs = [];
	
	if(typeof(deployJava) != "undefined")
	{
		JVMs = deployJava.getJREs();
	}
	else
	{
		JVMs = INFO.DeployTK_versions;
	}
	
	if(runningJavaVersion == 'Unknown' || runningJavaVersion == null || runningJavaVersion == "wait")
	{		
		
		if(JVMs.length > 0)
		{
			//MAC
			if(BrowserDetect.OS == 'Mac')
			{
				if(compareVersion(JVMs[0],requiredJavaVersion)==true)
				{
					return messageMultipleJVMs;
				}
				else
				{
					return messageMissingCorrectJVM;
				}
			}
			else
			{
				if(JVMs.length > 1)
				{
					if(IsCorrectVersionInstalled(JVMs))
					{
						return messageMultipleJVMs;
					}
					else
					{
						return messageMissingCorrectJVM;
					}
				}
				else
				{
					if(compareVersion(JVMs[0],requiredJavaVersion)==true)
					{
						return messageMultipleJVMs;
						//correctVersion = true;
					}
					else
					{
						return messageMissingCorrectJVM;
					}
				}
			}
					
		}
		else
		{		
			if(typeof(deployJava) != "undefined" && BrowserDetect.browser != 'Opera')
			{
				install = setTimeout("deployJava.installLatestJRE()", 20000);
				return messNoJavaInstalledDeploy;
			}
			else
			{
				return messNoJavaInstalledNoDeploy;
			}
		}
	}
	else if(compareVersion(runningJavaVersion,requiredJavaVersion)==true)
	{
		correctVersion = true;
		return '';
	}
	else
	{
		if(BrowserDetect.OS == 'Mac')
		{
			
			if(JVMs.length > 0)
			{
				if(runningJavaVersion != JVMs[0])
				{
					if(compareVersion(JVMs[0],requiredJavaVersion)==true)
					{
						return messageMultipleJVMs;
					}
					else
					{
						return messageMissingCorrectJVM;
					}
				}
				else
				{
					return messageMissingCorrectJVM;
				}
			}
			else
			{	
				return messageMissingCorrectJVM;
			}
		}
		else
		{
			if(JVMs.length > 1)
			{
				if(IsCorrectVersionInstalled(JVMs)==true)
				{
					return messageMultipleJVMs;
				}
				else
				{
					return messageMissingCorrectJVM;
				}
			}
			else if(runningJavaVersion != JVMs[0]){
					
					if(compareVersion(JVMs[0],requiredJavaVersion)==true)
					{
						return messageMultipleJVMs;
					}
					else
					{
						return messageMissingCorrectJVM;
					}
			}
			else
			{
				return messageMissingCorrectJVM;
				//return messageMultipleJVMs;
				//return messNoJavaInstalledNoDeploy;
				//return messNoJavaInstalledDeploy;
			}
		}
	}	
}
