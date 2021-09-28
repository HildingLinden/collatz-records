	GLOBAL	?collatz@@YAFPEA_K_J@Z, _Z7collatzPml

	section	.text
_Z7collatzPml:	; linux entry, move parameters to windows rules
	mov rcx, rdi
	mov rdx, rsi

?collatz@@YAFPEA_K_J@Z: ; windows entry

	mov r8, 6148914691236517204 ; UINT64_t / 3 - 1
	mov	rax, [rcx]			; n = j
	xor	r9, r9				; steps = 0

loop1:
	cmp rax, r8				; Will not go over UINT64_t if if the number is even but that would require quite a few extra instructions to check
	jg	exit				; break loop if overflow
	
	lea	r10, [rax+rax*2]	; tmp = 3 * n	
	add r10, 1				; tmp += 1
	
	add	r9, 1				; steps++
	
	shr	rax, 1				; n /= 2
	cmovc	rax, r10		; if n%2!=0 n = tmp
	
	cmp	rax, rdx			; check if n > LUT.size()
	jg 	loop1				; loop if true

exit:
	mov [rcx], rax			; save the number to the reference parameter
	mov	rax, r9				; return steps
	
	ret
