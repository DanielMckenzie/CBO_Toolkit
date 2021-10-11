      SUBROUTINE GROUP ( GVALUE, LGVALU, FVALUE, GPVALU, NCALCG, 
     *                   ITYPEG, ISTGPA, ICALCG, LTYPEG, LSTGPA, 
     *                   LCALCG, LFVALU, LGPVLU, DERIVS, IGSTAT )
      INTEGER LGVALU, NCALCG, LTYPEG, LSTGPA
      INTEGER LCALCG, LFVALU, LGPVLU, IGSTAT
      LOGICAL DERIVS
      INTEGER ITYPEG(LTYPEG), ISTGPA(LSTGPA), ICALCG(LCALCG)
      DOUBLE PRECISION GVALUE(LGVALU,3), FVALUE(LFVALU), GPVALU(LGPVLU)
C
C  Problem name : ALLINITC  
C
C  -- produced by SIFdecode 1.0
C
      INTEGER IGRTYP, IGROUP, IPSTRT, JCALCG
      DOUBLE PRECISION GVAR  
      IGSTAT = 0
      DO     3 JCALCG = 1, NCALCG
       IGROUP = ICALCG(JCALCG)
       IGRTYP = ITYPEG(IGROUP)
       IF ( IGRTYP == 0 ) GO TO     3
       IPSTRT = ISTGPA(IGROUP) - 1
       GO TO (    1,    2
     *                                                        ), IGRTYP
C
C  Group type : TRIVIAL 
C
    1  CONTINUE 
       GVAR  = FVALUE(IGROUP)
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= GVAR                                     
       ELSE
        GVALUE(IGROUP,2)= 1.0                                      
        GVALUE(IGROUP,3)= 0.0                                      
       END IF
       GO TO     3
C
C  Group type : L2      
C
    2  CONTINUE 
       GVAR  = FVALUE(IGROUP)
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= GVAR * GVAR                              
       ELSE
        GVALUE(IGROUP,2)= GVAR + GVAR                              
        GVALUE(IGROUP,3)= 2.0                                      
       END IF
    3 CONTINUE
      RETURN
      END
