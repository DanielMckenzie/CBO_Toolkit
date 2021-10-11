      SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, NCALCF, ITYPEE, 
     *                   ISTAEV, IELVAR, INTVAR, ISTADH, ISTEPA, 
     *                   ICALCF, LTYPEE, LSTAEV, LELVAR, LNTVAR, 
     *                   LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, 
     *                   LEPVLU, IFFLAG, IFSTAT )
      INTEGER NCALCF, IFFLAG, LTYPEE, LSTAEV, LELVAR, LNTVAR
      INTEGER LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, LEPVLU
      INTEGER IFSTAT
      INTEGER ITYPEE(LTYPEE), ISTAEV(LSTAEV), IELVAR(LELVAR)
      INTEGER INTVAR(LNTVAR), ISTADH(LSTADH), ISTEPA(LSTEPA)
      INTEGER ICALCF(LCALCF)
      DOUBLE PRECISION FUVALS(LFVALU), XVALUE(LXVALU), EPVALU(LEPVLU)
C
C  Problem name : NGONE     
C
C  -- produced by SIFdecode 1.0
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION DX    , SY    , U     , XA    , XB    
      DOUBLE PRECISION YA    , YB    , ZA    , ZB    
      IFSTAT = 0
      DO     3 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2
     *                                                        ), IELTYP
C
C  Element type : 2PR2I     
C
    1  CONTINUE
       XA     = XVALUE(IELVAR(ILSTRT+     1))
       XB     = XVALUE(IELVAR(ILSTRT+     2))
       YA     = XVALUE(IELVAR(ILSTRT+     3))
       YB     = XVALUE(IELVAR(ILSTRT+     4))
       DX     =   XA    
     *          - XB    
       SY     =   YA    
     *          + YB    
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= DX * SY                                  
       ELSE
        FUVALS(IGSTRT+     1)= SY                                       
        FUVALS(IGSTRT+     2)= DX                                       
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=1.0                                      
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO     3
C
C  Element type : ISQ       
C
    2  CONTINUE
       ZA     = XVALUE(IELVAR(ILSTRT+     1))
       ZB     = XVALUE(IELVAR(ILSTRT+     2))
       U      =   ZA    
     *          - ZB    
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= U * U                                    
       ELSE
        FUVALS(IGSTRT+     1)= U + U                                    
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0                                      
        END IF
       END IF
    3 CONTINUE
      RETURN
      END
