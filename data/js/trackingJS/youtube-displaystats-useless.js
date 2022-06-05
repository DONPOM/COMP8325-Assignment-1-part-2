(function() {
  var creativeDefinition = {
    customScriptUrl: '',
    isDynamic: true,
    delayedImpression: false,
    standardEventIds: {
      DISPLAY_TIMER: '2',
      INTERACTION_TIMER: '3',
      INTERACTIVE_IMPRESSION: '4',
      FULL_SCREEN_VIDEO_PLAYS: '5',
      FULL_SCREEN_VIDEO_COMPLETES: '6',
      FULL_SCREEN_AVERAGE_VIEW_TIME: '7',
      MANUAL_CLOSE: '8',
      BACKUP_IMAGE_IMPRESSION: '9',
      EXPAND_TIMER: '10',
      VIDEO_PLAY: '11',
      VIDEO_VIEW_TIMER: '12',
      VIDEO_COMPLETE: '13',
      VIDEO_INTERACTION: '14',
      VIDEO_PAUSE: '15',
      VIDEO_MUTE: '16',
      VIDEO_REPLAY: '17',
      VIDEO_MIDPOINT: '18',
      FULL_SCREEN_VIDEO: '19',
      VIDEO_STOP: '20',
      VIDEO_FIRST_QUARTILE: '960584',
      VIDEO_THIRD_QUARTILE: '960585',
      VIDEO_UNMUTE: '149645',
      FULL_SCREEN: '286263',
      DYNAMIC_CREATIVE_IMPRESSION: '536393',
      HTML5_CREATIVE_IMPRESSION: '871060'
    },
    exitEvents: [
      {
        name: 'YT_TSS_EXT: BG Click',
        reportingId: '2103342',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11408945\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT: Toyota Logo',
        reportingId: '2103315',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11408945\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT: View All Special Offers',
        reportingId: '2103330',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11408945\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_AurionAT-X: BG Click',
        reportingId: '2103381',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750303\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_AurionAT-X: Book a Test Drive',
        reportingId: '2103349',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709988\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_AurionAT-X: Download a Brochure',
        reportingId: '2103348',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709987\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_AurionAT-X: Find out more',
        reportingId: '2103387',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750303\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_CamryRZ: BG Click',
        reportingId: '2103335',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750306\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_CamryRZ: Book a Test Drive',
        reportingId: '2103358',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709978\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_CamryRZ: Download a Brochure',
        reportingId: '2103389',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709977\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_CamryRZ: Find out more',
        reportingId: '2103366',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750306\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_CorollaAscentSportHatch: BG Click',
        reportingId: '2103380',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750305\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_CorollaAscentSportHatch: Book a Test Drive',
        reportingId: '2103393',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709982\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_CorollaAscentSportHatch: Download a Brochure',
        reportingId: '2103354',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709981\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_CorollaAscentSportHatch: Find out more',
        reportingId: '2103326',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750305\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_HiLux4x4SR5Double Cab: BG Click',
        reportingId: '2103394',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750300\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_HiLux4x4SR5Double Cab: Book a Test Drive',
        reportingId: '2103399',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709992\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_HiLux4x4SR5Double Cab: Download a Brochure',
        reportingId: '2103386',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709991\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_HiLux4x4SR5Double Cab: Find out more',
        reportingId: '2103372',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750300\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_LC70SCCCWorkMate: BG Click',
        reportingId: '2103384',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750299\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_LC70SCCCWorkMate: Book a Test Drive',
        reportingId: '2103375',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709996\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_LC70SCCCWorkMate: Download a Brochure',
        reportingId: '2103353',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709995\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_LC70SCCCWorkMate: Find out more',
        reportingId: '2103400',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750299\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_PradoGXL: BG Click',
        reportingId: '2103378',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750302\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_PradoGXL: Book a Test Drive',
        reportingId: '2103368',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709994\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_PradoGXL: Download a Brochure',
        reportingId: '2103336',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709993\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_PradoGXL: Find out more',
        reportingId: '2103340',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750302\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_PriusC: BG Click',
        reportingId: '2103331',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750304\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_PriusC: Book a Test Drive',
        reportingId: '2103385',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709986\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_PriusC: Download a Brochure',
        reportingId: '2103312',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709985\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_PriusC: Find out more',
        reportingId: '2103383',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750304\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_Rav4AWDRange: BG Click',
        reportingId: '2103382',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750301\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_Rav4AWDRange: Book a Test Drive',
        reportingId: '2103371',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709980\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_Rav4AWDRange: Download a Brochure',
        reportingId: '2103402',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11709979\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      },
      {
        name: 'YT_TSS_EXT_Rav4AWDRange: Find out more',
        reportingId: '2103338',
        url: 'https://bs.serving-sys.com/BurstingPipe/adServer.bs?cn\x3dtf\x26c\x3d20\x26mc\x3dclick\x26pli\x3d11750301\x26PluID\x3d0\x26ord\x3d%n',
        targetWindow: '_blank',
        windowProperties: ''
      }
    ],
    timerEvents: [
      {
        name: 'YT_TSS_Aurion AT-X_TMR: View Video Timer',
        reportingId: '2103376',
        videoData: null
      },
      {
        name: 'YT_TSS_Camry RZ_TMR: View Video Timer',
        reportingId: '2103327',
        videoData: null
      },
      {
        name: 'YT_TSS_Corolla Ascent Sport Hatch_TMR: View Video Timer',
        reportingId: '2103391',
        videoData: null
      },
      {
        name: 'YT_TSS_Generic_TMR: View Video Timer',
        reportingId: '2103334',
        videoData: null
      },
      {
        name: 'YT_TSS_HiLux 4x4 SR5 Double Cab_TMR: View Video Timer',
        reportingId: '2103332',
        videoData: null
      },
      {
        name: 'YT_TSS_LC70 SCCC WorkMate_TMR: View Video Timer',
        reportingId: '2103320',
        videoData: null
      },
      {
        name: 'YT_TSS_Prado GXL_TMR: View Video Timer',
        reportingId: '2103379',
        videoData: null
      },
      {
        name: 'YT_TSS_Prius C_TMR: View Video Timer',
        reportingId: '2103317',
        videoData: null
      },
      {
        name: 'YT_TSS_RAV4 AWD Range_TMR: View Video Timer',
        reportingId: '2103357',
        videoData: null
      }
    ],
    counterEvents: [
      {
        name: 'Close Ad',
        reportingId: '1041107',
        videoData: null
      },
      {
        name: 'YT_TSS_CTR: + Camry RZ',
        reportingId: '2103356',
        videoData: null
      },
      {
        name: 'YT_TSS_CTR: + Corolla Ascent Sport Manual Hatch',
        reportingId: '2103319',
        videoData: null
      },
      {
        name: 'YT_TSS_CTR: + Rav4',
        reportingId: '2103395',
        videoData: null
      },
      {
        name: 'YT_TSS_CTR: Close Ad',
        reportingId: '2103329',
        videoData: null
      },
      {
        name: 'YT_TSS_CTR: Replay Animation',
        reportingId: '2103701',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Aurion AT-X: Change Quality',
        reportingId: '2103352',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Aurion AT-X: Full Screen',
        reportingId: '2103397',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Aurion AT-X: Play',
        reportingId: '2103398',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Aurion AT-X: Video_0_Percent',
        reportingId: '2103360',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Aurion AT-X: Video_100_Percent',
        reportingId: '2103359',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_CamryRZ: Change Quality',
        reportingId: '2103346',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_CamryRZ: Full Screen',
        reportingId: '2103313',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_CamryRZ: Play',
        reportingId: '2103390',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_CamryRZ: Video_0_Percent',
        reportingId: '2103367',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_CamryRZ: Video_100_Percent',
        reportingId: '2103364',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Corolla Ascent Sport Hatch: Change Quality',
        reportingId: '2103323',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Corolla Ascent Sport Hatch: Full Screen',
        reportingId: '2103341',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Corolla Ascent Sport Hatch: Play',
        reportingId: '2103343',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Corolla Ascent Sport Hatch: Video_0_Percent',
        reportingId: '2103350',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Corolla Ascent Sport Hatch: Video_100_Percent',
        reportingId: '2103396',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Generic: Change Quality',
        reportingId: '2103347',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Generic: Full Screen',
        reportingId: '2103369',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Generic: Play',
        reportingId: '2103321',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Generic: Video_0_Percent',
        reportingId: '2103325',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Generic: Video_100_Percent',
        reportingId: '2103361',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_HiLux 4x4 SR5 Double Cab: Change Quality',
        reportingId: '2103388',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_HiLux 4x4 SR5 Double Cab: Full Screen',
        reportingId: '2103328',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_HiLux 4x4 SR5 Double Cab: Play',
        reportingId: '2103324',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_HiLux 4x4 SR5 Double Cab: Video_0_Percent',
        reportingId: '2103344',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_HiLux 4x4 SR5 Double Cab: Video_100_Percent',
        reportingId: '2103345',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_LC70 SCCC WorkMate: Change Quality',
        reportingId: '2103333',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_LC70 SCCC WorkMate: Full Screen',
        reportingId: '2103362',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_LC70 SCCC WorkMate: Play',
        reportingId: '2103392',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_LC70 SCCC WorkMate: Video_0_Percent',
        reportingId: '2103370',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_LC70 SCCC WorkMate: Video_100_Percent',
        reportingId: '2103403',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Prado GXL: Change Quality',
        reportingId: '2103374',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Prado GXL: Full Screen',
        reportingId: '2103377',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Prado GXL: Play',
        reportingId: '2103316',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Prado GXL: Video_0_Percent',
        reportingId: '2103339',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Prado GXL: Video_100_Percent',
        reportingId: '2103355',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Prius C: Change Quality',
        reportingId: '2103401',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Prius C: Full Screen',
        reportingId: '2103311',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Prius C: Play',
        reportingId: '2103373',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Prius C: Video_0_Percent',
        reportingId: '2103351',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_Prius C: Video_100_Percent',
        reportingId: '2103363',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_RAV4 AWD Range: Change Quality',
        reportingId: '2103318',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_RAV4 AWD Range: Full Screen',
        reportingId: '2103337',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_RAV4 AWD Range: Play',
        reportingId: '2103365',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_RAV4 AWD Range: Video_0_Percent',
        reportingId: '2103314',
        videoData: null
      },
      {
        name: 'YT_TSS_Video_RAV4 AWD Range: Video_100_Percent',
        reportingId: '2103322',
        videoData: null
      }
    ],
    childFiles: [
      {
        name: 'TOY10799OD_Masthead_v1.swf',
        url: '/ads/richmedia/studio/pv2/34884991/20141122181325157/TOY10799OD_Masthead_v1.swf',
        isVideo: false,
        transcodeInformation: null
      },
      {
        name: 'TOY10799OD_YouTube_Masthead_BUG 2.jpg',
        url: '/ads/richmedia/studio/pv2/34884991/20141122181325157/TOY10799OD_YouTube_Masthead_BUG 2.jpg',
        isVideo: false,
        transcodeInformation: null
      },
      {
        name: 'TOY10799OD_Additional_v1.swf',
        url: '/ads/richmedia/studio/pv2/34884991/20141122181325157/TOY10799OD_Additional_v1.swf',
        isVideo: false,
        transcodeInformation: null
      }
    ],
    videoFiles: [
    ],
    videoEntries: [
    ],
    primaryAssets: [
      {
        id: '34978323',
        artworkType: 'FLASH',
        displayType: 'BANNER',
        width: '970',
        height: '250',
        servingPath: '/ads/richmedia/studio/pv2/34884991/20141122181325157/main_polite_INP.swf',
        zIndex: '1000000',
        customCss: '',
        flashArtworkTypeData: {
          actionscriptVersion: '3',
          wmode: 'opaque',
          sdkVersion: '2.4.4',
          flashBackgroundColor: '',
          allowScriptAccess: 'always'
        },
        htmlArtworkTypeData: null,
        floatingDisplayTypeData: null,
        expandingDisplayTypeData: null,
        imageGalleryTypeData: null,
        pageSettings:null,
layoutsConfig: null,
layoutsApi: null
      }
    ]
  }
  var rendererDisplayType = '';
  rendererDisplayType += 'flash_';
  var rendererFormat = 'inpage';
  var rendererName = rendererDisplayType + rendererFormat;

  var creativeId = '60468343';
  var adId = '286498825';
  var templateVersion = '200_58';
  var studioObjects = window['studioV2'] = window['studioV2'] || {};
  var creativeObjects = studioObjects['creatives'] = studioObjects['creatives'] || {};
  var creativeKey = [creativeId, adId].join('_');
  var creative = creativeObjects[creativeKey] = creativeObjects[creativeKey] || {};
  creative['creativeDefinition'] = creativeDefinition;
  var adResponses = creative['adResponses'] || [];
  for (var i = 0; i < adResponses.length; i++) {
    adResponses[i].creativeDto && adResponses[i].creativeDto.csiEvents &&
        (adResponses[i].creativeDto.csiEvents['pe'] =
            adResponses[i].creativeDto.csiEvents['pe'] || (+new Date));
  }
  var loadedLibraries = studioObjects['loadedLibraries'] = studioObjects['loadedLibraries'] || {};
  var versionedLibrary = loadedLibraries[templateVersion] = loadedLibraries[templateVersion] || {};
  var typedLibrary = versionedLibrary[rendererName] = versionedLibrary[rendererName] || {};
  if (typedLibrary['bootstrap']) {
    typedLibrary.bootstrap();
  }
})();
