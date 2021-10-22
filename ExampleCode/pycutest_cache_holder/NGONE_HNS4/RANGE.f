      SUBROUTINE RANGE( IELEMN, TRANSP, W1, W2, nelvar, ninvar,
     *                  itype, LW1, LW2 )
      INTEGER IELEMN, nelvar, ninvar, itype, LW1, LW2
      LOGICAL TRANSP
      DOUBLE PRECISION W1( LW1 ), W2( LW2 )
C
C  Problem name : NGONE     
C
C  -- produced by SIFdecode 1.0
C
C  TRANSP = .FALSE. <=> W2 = U * W1
C  TRANSP = .TRUE.  <=> W2 = U(transpose) * W1
C
      INTEGER I
      GO TO (    1,    2
     *                                                        ), ITYPE
C
C  Element type : 2PR2I     
C
    1 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) = - W1(     1 ) 
         W2(     3 ) =   W1(     2 ) 
         W2(     4 ) =   W1(     2 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) 
     *                 - W1(     2 ) 
         W2(     2 ) =   W1(     3 ) 
     *                 + W1(     4 ) 
      END IF
      RETURN
C
C  Element type : ISQ       
C
    2 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) = - W1(     1 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) 
     *                 - W1(     2 ) 
      END IF
      RETURN
      END